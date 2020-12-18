# SenseEarth2020 - ChangeDetection

## Task Introduction




## Our Method

**The core practice is using self-distillation strategy to assign pseudo labels to unchanged areas.**

Specifically, in our experiments, predictions of five HRNet-based segmentation models are ensembled, 
serving as pseudo labels of unchanged areas. 

The overall training process can be summarized as:

* Train several large segmentation models.
* Ensemble their predictions on unchanged areas.
* Train a small segmentation with both labeled and pseudo-labeled areas.


