# SenseEarth2020 - ChangeDetection

## Task Introduction

![](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection/blob/master/docs/pipeline.pdf)


## Our Method

**The core practice is using self-distillation strategy to assign pseudo labels to unchanged areas.**

Specifically, in our experiments, predictions of five HRNet-based segmentation models are ensembled, 
serving as pseudo labels of unchanged areas. 

The overall training process can be summarized as:

* Training several large segmentation models.
* Ensembling their predictions on unchanged areas.
* Training a smaller model with both labeled and pseudo labeled areas.


For more details, please refer to the 
[technical report](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection/blob/master/docs/technical%20report.pdf) 
and [presentation](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection/blob/master/docs/presentation.pptx).
