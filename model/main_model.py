import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


from .models import New_Audio_Guided_Attention
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention
from .Dual_lstm import Dual_lstm
import torch.nn.functional as F

save_mats = dict()

class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class + 1, kernel_size=1, stride=1, padding=0,
                      bias=False)
        )

    def forward(self, content):
        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model=d_model, num_class=28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class weak_main_model(nn.Module):
    def __init__(self, config):
        super(weak_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = self.config["video_inputdim"]
        self.video_fc_dim = self.config["video_inputdim"]
        self.d_model = self.config["d_model"]
        self.audio_input_dim = self.config["audio_inputdim"]
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model,
                                                            feedforward_dim=2048)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.audio_input_dim, d_model=self.d_model,
                                                            feedforward_dim=2048)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.audio_input_dim, d_model=self.d_model,
                                                         feedforward_dim=1024)
        # self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=512, d_model=256, num_layers=1)
        self.AVInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        # self.localize_module = WeaklyLocalizationModule(self.d_model)
        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),

            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.CAS_model = CAS_Module(d_model=self.d_model, num_class=28)
        self.classifier = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.audio_cas = nn.Linear(self.d_model, 29)
        self.video_cas = nn.Linear(self.d_model, 29)
        self.video_fc = nn.Linear(self.d_model, out_features=29)
        self.audio_fc = nn.Linear(self.d_model, out_features=29)
        self.softmax = nn.Softmax(dim=-1)
        self.index = 0

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # this fc is optinal, that is used for adaption of different visual features (e.g., vgg, resnet).
        # audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        # visual_rnn_input = visual_feature

        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        # 音频自注意力块
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        # video自注意力
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)
        # 两个门
        audio_gate = self.audio_gated(video_key_value_feature)
        video_gate = self.video_gated(audio_key_value_feature)

        av_gate = (audio_gate + video_gate) / 2
        av_gate = av_gate.permute(1, 0, 2)

        video_query_output = (1 - self.alpha) * video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = (1 - self.alpha) * audio_query_output + video_gate * audio_query_output * self.alpha #[10,64,256]

        #弱监督kl
        # video_fc = self.video_fc(video_query_output)  # [10, 64, 29]
        # audio_fc = self.audio_fc(audio_query_output)
        # video_fc = self.relu(video_query_output)
        # audio_fc = self.relu(audio_query_output)
        # video_sim = self.softmax(video_fc)
        # audio_sim = self.softmax(audio_fc)
        # kl_loss = F.kl_div(audio_sim.log(), video_sim, reduction='sum')
        # # TODO
        # # kl_loss = kl_loss.item()
        # if self.index % 5 == 0:
        #     print("kl_loss: ", kl_loss)
        # self.index += 1

        video_cas = self.video_cas(video_query_output)
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)

        video_cas_gate = video_cas.sigmoid()
        audio_cas_gate = audio_cas.sigmoid()
        #
        # video_cas_gate = (video_cas_gate > 0.01).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.01).float()*audio_cas_gate

        # video_cas = audio_cas_gate.unsqueeze(1) * video_cas
        # audio_cas = video_cas_gate.unsqueeze(1) * audio_cas
        #
        # sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        # topk_scores_video = sorted_scores_video[:, :4, :]
        # score_video = torch.mean(topk_scores_video, dim=1)
        # sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        # topk_scores_audio = sorted_scores_audio[:, :4, :]
        # score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 29]
        #
        # video_cas_gate = score_video.sigmoid()
        # audio_cas_gate = score_audio.sigmoid()
        # video_cas_gate = (video_cas_gate > 0.5).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.5).float()*audio_cas_gate

        #
        # av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)
        # scores = self.localize_module((video_query_output+audio_query_output)/2)

        fused_content = (video_query_output + audio_query_output) / 2
        # fused_content = video_query_output
        fused_content = fused_content.transpose(0, 1)
        # is_event_scores = self.classifier(fused_content)

        cas_score = self.CAS_model(fused_content)
        # cas_score = cas_score + 0.2*video_cas_gate.unsqueeze(1)*cas_score + 0.2*audio_cas_gate.unsqueeze(1)*cas_score
        cas_score = self.gamma * video_cas_gate * cas_score + self.gamma * audio_cas_gate * cas_score
        # cas_score = cas_score*2
        sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :4, :]
        raw_logits = torch.mean(topk_scores, dim=1)[:, None, :]  # [32, 29]

        # fused_logits = is_event_scores.sigmoid() * raw_logits
        fused_logits = av_gate * raw_logits
        # fused_scores, _ = fused_logits.sort(descending=True, dim=1)
        # topk_scores = fused_scores[:, :3, :]
        # logits = torch.mean(topk_scores, dim=1)
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        event_scores = event_scores

        return av_gate.squeeze(), raw_logits.squeeze(), event_scores


class supv_main_model(nn.Module):
    def __init__(self, config):
        super(supv_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = self.config['video_inputdim']
        self.audio_input_dim = self.config['audio_inputdim']
        # self.index = 0
        self.video_fc_dim = 512
        self.d_model = self.config['d_model']

        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim,
                                                 d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model)
        self.video_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.video_fc = nn.Linear(self.d_model,out_features=28)
        self.audio_fc = nn.Linear(self.d_model,out_features=28)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']
        self.index =0
        self.avg_loss_array = torch.tensor([])
    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]

        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)
        #抑制前音频
        save_mats["before_audio_q"] = audio_query_output.detach().cpu().numpy()
        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)
        #抑制前结果
        save_mats["before_video_q"] = video_query_output.detach().cpu().numpy()
        # video_key_value_feature = torch.exp(video_key_value_feature)
        # audio_key_value_feature = torch.exp(audio_key_value_feature)

        audio_gate = self.audio_gated(audio_key_value_feature)#(10,64,1)
        #打印输出时间抑制门
        # save_mats["AudioGate"] = audio_gate.detach().cpu().numpy()



        video_gate = self.video_gated(video_key_value_feature)  #(10,64,1)
        #打印输出
        # save_mats["VideoGate"] = video_gate.detach().cpu().numpy()

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha  # [10, 64, 256]
        save_mats["after_video_q"] = video_query_output.detach().cpu().numpy() #[10,64,256]
        save_mats["after_audio_q"] = audio_query_output.detach().cpu().numpy()
        # 加入kl散度
        video_fc = self.video_fc(video_query_output)# [10, 64, 28]
        audio_fc = self.audio_fc(audio_query_output)
        video_fcc = self.relu(video_fc)
        audio_fcc = self.relu(audio_fc)
        video_sim = self.softmax(video_fcc)
        audio_sim = self.softmax(audio_fcc)
        kl_loss = F.kl_div(audio_sim.log(), video_sim, reduction='sum')


        #事件级背景抑制
        video_cas = self.video_cas(video_query_output)  # [10, 64, 28]
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)

        #不排序
        # _,count_fg = k_count(labels)
        # fg = 1 - count_fg * 1
        # fg = torch.unsqueeze(fg,dim=-1)
        # fg_video_result = fg * video_cas
        # fg_audio_result = fg * audio_cas
        #
        # fg_sum = torch.sum(fg,dim=1) + 1e-10
        # fg_video_result_sum = torch.sum(fg_video_result,dim=1)
        # fg_audio_result_sum = torch.sum(fg_audio_result,dim=1)
        #
        # fg_video_res = fg_video_result_sum / fg_sum
        # fg_audio_res = fg_audio_result_sum / fg_sum


        #排序结果
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        #打印输出k的结果
        save_mats["k_sort_video"] = sorted_scores_video.detach().cpu().numpy() #[64,10,28]
        save_mats["k_sort_audio"] = sorted_scores_audio.detach().cpu().numpy()
        # fg_video_order = fg *  sorted_scores_video
        # fg_audio_order = fg * sorted_scores_audio
        # fg_video_order_sum = torch.sum(fg_video_order,dim=1)
        # fg_audio_order_sum = torch.sum(fg_audio_order,dim=1)
        #
        # score_video = fg_video_order_sum / fg_sum
        # score_audio = fg_audio_order_sum / fg_sum

        # batchsize = sorted_scores_video.size(0)
        # score_video = torch.zeros(batchsize, 28).cuda()
        # score_audio = torch.zeros(batchsize, 28).cuda()
        # counts,_ = k_count(labels)
        # for i in range(len(counts)):
        #     k = counts[i].item()
        #     topk_scores_video = sorted_scores_video[i][:k, :]
        #     topk_scores_audio = sorted_scores_audio[i][:k, :]
        #     score_video[i] = torch.mean(topk_scores_video, dim=0)
        #     score_audio[i] = torch.mean(topk_scores_audio,dim=0)


        topk_scores_video = sorted_scores_video[:, :4, :]  #[64,4,28]
        score_video = torch.mean(topk_scores_video, dim=1) #[64,28]
        #打印事件抑制结果
        # save_mats["Score_video"] = score_video.detach().cpu().numpy()
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [64, 28]
        # 打印事件抑制结果
        #         # save_mats["Scre_audio"] = score_audio.detach().cpu().numpy()



        # event_visual_gate = score_video.sigmoid()
        # event_audio_gate = score_audio.sigmoid()

        # v_score = (score_video + fg_video_res) / 2
        # a_score = (score_audio + fg_audio_res ) / 2
        # av_score = (v_score + a_score) / 2
        av_score = (score_video + score_audio) / 2
        # av_score = (av_score_disorder + av_score) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        #is_event_scores [10,64,1],event_scores[64,28]
        is_event_scores, event_scores = self.localize_module((video_query_output + audio_query_output) / 2)
        a_v = (video_query_output + audio_query_output) / 2 #[10,64,256]

        #打印结果
        save_mats["fusion_feature"] = a_v.detach().cpu().numpy()
        # 打印每秒预测结果结果
        # is_event_scores_sigmoid = is_event_scores.sigmoid()
        # save_mats["is_event_scores"] = is_event_scores_sigmoid.detach().cpu().numpy()

        #原始整体loss
        event_scores = event_scores + self.gamma * av_score

        return is_event_scores, event_scores, audio_visual_gate, av_score,kl_loss

def k_count(labels):
    _, targets = labels.max(-1)
    count_bg = torch.sum(torch.eq(targets, 28), dim=1)
    count_fg = torch.eq(targets, 28)
    count_foreground = 10 - count_bg
    return count_foreground,count_fg