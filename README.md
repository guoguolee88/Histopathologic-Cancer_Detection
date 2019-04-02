# Histopathologic-Cancer-Detection
Kaggle Competition: Identify metastatic tissue in histopathologic scans of lymph node sections

## Data
- PatchCamelyon (PCam)

## Quick Start
- convert .tif to .png
- split dataset into train, val
- create tfrecord file
- execute train.py

## Evaluation
- execute eval.py

## Done
- Data split applied
  - data class balancing
  - WSI (Whole slide imaging)
- Random augmentation from imgaug package: flips, rotations, crops, saturation
- Using tfrecord.
- Val/Test TTA
- Applied SENet (from https://github.com/kobiso/SENet-tensorflow-slim)

## TODO 
- ensemble module (with 5 or 10 models trained on different subsets).
  - ex) ensemble of 5 se_resnet50 models.
- Save model every time AUROC is increased. By the end of the training save the best model.
  AUC = AUROC (the Area Under a ROC Curve)
- Apply ReduceLROnPlateau
    : learning rate to be reduced when training is not progressing.
- Train the added layers for 1-2 epoch and then the whole network for another 3-4 ??

## References from
- https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
- https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
- https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/84790
- https://github.com/kobiso/SENet-tensorflow-slim
