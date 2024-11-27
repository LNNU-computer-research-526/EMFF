# Event-Level Multimodal Feature Fusion for Audio-Visual Event Localization

### How to run the code
### Step 1: Check the compatibility of your python packages, we recommend you to use the following setting that has passed our test if you want to reproduce the results.
```python
Python ==  3.9
Pytorch ==  1.11.0
CUDA ==  11.4
h5py ==  3.1.0
numpy ==  1.21.5
```


### Step 2: Put in the dataset
The VGG visual features can be downloaded from [Visual_feature](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing).

The VGG-like audio features can be downloaded from [Audio_feature](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing).

The noisy visual features used for weakly-supervised setting can be downloaded from [Noisy_visual_feature](https://drive.google.com/file/d/1I3OtOHJ8G1-v5G2dHIGCfevHQPn-QyLh/view?usp=sharing).

After downloading the features, please place them into the `data` folder.

If you are interested in the AVE raw videos, please refer to this [repo](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK) and download the AVE dataset. 

### Step 3:Training and Evaluating EMFF

#### Fully-Supervised Setting
The `configs/main.json` contains the main hyper-parameters used for fully-supervised training.

Training 
```bash
bash supv_train.sh
```
Evaluating

```bash
bash supv_test.sh
```


## Pretrained model
The pretrained models is in folder `Exps` .

You can try different parameters or random seeds if you want to retrain the model, the results may be better.

## Acknowledgement

Part of our code is borrowed from the following repositories.

- [YapengTian/AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18)
- [CMRAN](https://github.com/FloretCat/CMRAN)
- [CMBS](https://github.com/marmot-xy/CMBS/tree/main)


We would like to thank the authors for releasing their codes. Please also consider citing their works.

