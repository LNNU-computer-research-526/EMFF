import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.main_model import supv_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset
import torch.nn.functional as F
from model.main_model import save_mats
# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path = 'configs/main.json'
with open(config_path) as fp:
    config = json.load(fp)
print(config)


# =============================================================================

def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss


def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDataset('./data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('./data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    '''model setting'''
    mainModel = main_model(config['model'])
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            test_list.clear()
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_mats_to_file(epoch)
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                )
            print("-----------------------------")
            print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
            print("-----------------------------")
            test_list.clear()
        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    kl_losses = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    model.double()
    optimizer.zero_grad()


    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float precision
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()

        # is_event_scores, event_scores, audio_visual_gate, av_score = model(visual_feature, audio_feature)

        is_event_scores, event_scores, audio_visual_gate, av_score,kl_loss  = model(visual_feature, audio_feature)
        # is_event_scores, event_scores = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()
        audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze().contiguous()

        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        # _, labels_CAS = labels.max(-1)

        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        label_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())
        loss_cas = criterion_event(av_score, labels_event.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        # kl_loss = criterion_event(kl_loss, torch.zeros_like(kl_loss).cuda())
        # loss_cas_event = criterion_event(cas_out, labels_CAS.cuda())

        # loss = loss_is_event + label_is_gate + loss_event_class + loss_cas
        loss = loss_is_event + label_is_gate + loss_event_class + loss_cas+kl_loss

        # loss = loss_is_event + loss_event_class
        loss.backward()

        '''Compute Accuracy'''
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()


        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        kl_losses.update(kl_loss.item(), visual_feature.size(0) * 10)
        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Train_epoch_data/kl_loss', kl_losses.avg, epoch)
    logger.info(
            f'**************************************************************************\t'
            f"\tTrain results (acc): {train_acc.avg:.4f}%."
            f"\t results (kl_loss): {kl_losses.avg:.4f}%."

        )
    return losses.avg

test_list = []
@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()
    kl_losses = AverageMeter()
    model.eval()
    model.double()



    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float type
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        bs = visual_feature.size(0)
        is_event_scores, event_scores, audio_visual_gate, _ ,kl_loss= model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze()
        audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze()

        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = loss_is_event + loss_event_class + loss_is_gate+kl_loss

        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        accuracy.update(acc.item(), bs * 10)
        kl_losses.update(kl_loss.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)
        test_list.append(save_mats.copy())
        save_mats.clear()
        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
        f"\t results (kl_loss): {kl_losses.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    _, targets = labels.max(-1)
    # pos pred
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)  # foreground classification
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = 28  # 28 denotes bg
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc
    # # labels = labels[:, :, :-1]  # 28 denote background
    # _, targets = labels.max(-1)
    # # pos pred
    # is_event_scores = is_event_scores.sigmoid()
    # scores_pos_ind = is_event_scores > 0.5
    #
    # scores_mask = scores_pos_ind == 0
    #
    # _, event_class = event_scores.max(-1)  # foreground classification
    # pred = scores_pos_ind.long()
    # pred *= event_class[:, None]
    # # add mask
    # pred[scores_mask] = 28  # 28 denotes bg
    #
    # #
    # count_bg = torch.sum(torch.eq(targets, 28), dim=1)
    # count_foreground = 10 - count_bg
    # #below3
    # fg_id_3 = count_foreground <= 3
    # count_3 = torch.sum(fg_id_3, dim=0)
    # fg_id_33 = torch.unsqueeze(fg_id_3, dim=-1)
    # scores_pos_ind_3 = fg_id_33 * scores_pos_ind
    # pred_3 = scores_pos_ind_3.long()
    # #background
    # scores_mask_3 = scores_pos_ind_3 == 0
    # event_class_3 = fg_id_3 * event_class
    # targets_3 = fg_id_33 * targets
    # pred_3 *= event_class_3[:, None]
    # pred_3[scores_mask_3] = 28
    # correct_3 = pred_3.eq(targets_3)
    # correct_num_3 = correct_3.sum().double()
    # acc_3 = correct_num_3 * (100. / (count_3 * 10))
    # print("below3_acc",acc_3)
    #
    # #between4-7
    # fg_id_7 = (count_foreground > 3) & (count_foreground < 7)
    # count_7 = torch.sum(fg_id_7, dim=0)
    # fg_id_77 = torch.unsqueeze(fg_id_7, dim=-1)
    # scores_pos_ind_7 = fg_id_77 * scores_pos_ind
    # pred_7 = scores_pos_ind_7.long()
    # # background
    # scores_mask_7 = scores_pos_ind_7 == 0
    # event_class_7 = fg_id_7 * event_class
    # targets_7 = fg_id_77 * targets
    # pred_7 *= event_class_7[:, None]
    # pred_7[scores_mask_7] = 28
    # correct_7 = pred_7.eq(targets_7)
    # correct_num_7 = correct_7.sum().double()
    # acc_7 = correct_num_7 * (100. / (count_7 * 10))
    # print("between4~7_acc", acc_7)
    #
    # #more8
    # fg_id_8 =  count_foreground > 7
    # count_8 = torch.sum(fg_id_8, dim=0)
    # fg_id_88 = torch.unsqueeze(fg_id_8, dim=-1)
    # scores_pos_ind_8 = fg_id_88 * scores_pos_ind
    # pred_8 = scores_pos_ind_8.long()
    # # background
    # scores_mask_8 = scores_pos_ind_8 == 0
    # event_class_8 = fg_id_8 * event_class
    # targets_8 = fg_id_88 * targets
    # pred_8 *= event_class_8[:, None]
    # pred_8[scores_mask_8] = 28
    # correct_8 = pred_8.eq(targets_8)
    # correct_num_8 = correct_8.sum().double()
    # acc_8 = correct_num_8 * (100. / (count_8 * 10))
    # print("more_7_acc", acc_8)
    #
    #
    # correct = pred.eq(targets)
    # correct_num = correct.sum().double()
    # acc = correct_num * (100. / correct.numel())
    #
    # return acc


def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


def save_mats_to_file(epoch):
    for kindex,test_mats in enumerate(test_list):
        for k,mat in test_mats.items():
            with open(f"save_mats\Mat_{k}_Epoch_{epoch}_Sample_{kindex}.txt", "w") as f:
                mat_np = mat
                if len(mat_np.shape) > 2:
                    for j in mat_np:
                        np.savetxt(f, j, delimiter='\t', fmt='%.4f')
                        f.write("\n\n")
                else:
                    np.savetxt(f, mat_np, delimiter='\t', fmt='%.4f')
                f.close()









if __name__ == '__main__':
    main()

