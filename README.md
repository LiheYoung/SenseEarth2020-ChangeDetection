# SenseEarth2020 - ChangeDetection

**1st place in the Satellite Image Change Detection 
[Challenge](https://rs.sensetime.com/competition/index.html#/info) 
hosted by [SenseTime](https://www.sensetime.com/cn).**

## Our Method

### Task Description

Given two images of the same scene acquired at different times, we are required to mark the changed 
and unchanged areas. Moreover, as for the changed areas, we need to annotate their detailed semantic masks. 

The change detection task in this competition can be decomposed into two sub-tasks:
* binary segmentation of changed and unchanged areas.
* semantic segmentation of changed areas.

### Model

![image](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection/blob/master/docs/pipeline.png)

### Pseudo Labeling

**The core practice is using self-distillation strategy to assign pseudo labels to unchanged areas.**

Specifically, in our experiments, predictions of five HRNet-based segmentation models are ensembled, 
serving as pseudo labels of unchanged areas. 

The overall training process can be summarized as:

* Training multiple large segmentation models.
* Ensembling their predictions on unchanged areas.
* Training a smaller model with both labeled and pseudo labeled areas.

For more details, please refer to the 
[technical report](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection/blob/master/docs/technical%20report.pdf) 
and [presentation](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection/blob/master/docs/presentation.pptx).



## Getting Started

### Dataset
[Description](https://rs.sensetime.com/competition/index.html#/data) | [Download [password: f3qq]](https://pan.baidu.com/s/1Yg90vlAiKezSoxH7WEoV6g) 

### Pretrained Model
[HRNet-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HRNet-W40](https://1drv.ms/u/s!Aus8VCZ_C_33ck0gvo5jfoWBOPo) | [HRNet-W44](https://1drv.ms/u/s!Aus8VCZ_C_33czZQ0woUb980gRs) | [HRNet-W48](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) | [HRNet-W64](https://1drv.ms/u/s!Aus8VCZ_C_33gQbJsUPTIj3rQu99)

### File Organization
```
# store the whole dataset and pretrained backbones
mkdir -p data/dataset ; mkdir -p data/pretrained_models ;

# store the trained models
mkdir -p outdir/models ; 

# store the pseudo masks
mkdir -p outdir/masks/train/im1 ; mkdir -p outdir/masks/train/im2 ;

# store predictions of validation set and testing set
mkdir -p outdir/masks/val/im1 ; mkdir -p outdir/masks/val/im2 ;
mkdir -p outdir/masks/test/im1 ; mkdir -p outdir/masks/test/im2 ;

data
├── dataset                    # download from the link above
│   ├── train                  # training set
|   |   ├── im1
|   |   └── ...
│   └── val                    # the final testing set (without labels)
|
└── pretrained_models
    ├── hrnet_w18.pth
    ├── hrnet_w40.pth
    └── ...
```

### Training
```
# Please refer to utils/options.py for more arguments
# If hardware supports, more backbones can be trained, such as hrnet_w44, hrnet_w48
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone hrnet_w18 --pretrained --model pspnet --lightweight
```

### Pseudo Labeling & Re-training 
```
# This step is optional but important in performance improvement
# Modify the backbones, models and checkpoint paths in L20-40 in label.py manually according to your saved models
# It is better to ensemble multiple trained models for pseudo labeling

# Pseudo labeling
CUDA_VISIBLE_DEVICES=0,1,2,3 python label.py

# Re-training
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone hrnet_w18 --pretrained --model pspnet --lightweight --use-pseudo-label
```

### Testing
```
# Modify the backbones, models and checkpoint paths in L39-44 in test.py manually according to your saved models
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py
```

