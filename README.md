# Histopathologic-Cancer-Detection
Kaggle Competition: Identify metastatic tissue in histopathologic scans of lymph node sections

## Data
- PatchCamelyon (PCam)

## Quick Start
- convert .tif to .png
- create tfrecord file
- execute train.py

## Evaluation
- execute eval.py

## Data split applied
- data class balancing
- WSI (Whole slide imaging)

## TODO 
- make ensemble module (Will see how well it performs with 5 or 10 models).
- Save model every time AUROC is increased. By the end of the training save the best model.
  AUC = AUROC (the Area Under a ROC Curve) : ROC 커브의 밑면적을 구한 값이 바로 AUC. 이 값이 1에 가까울수록 성능이 좋다.
- Research and development the keywords below.
  - Apply ReduceLROnPlateau with the patience of 1-2 epocs
    : learning rate to be reduced when training is not progressing.
  - Train the added layers for 1-2 epoch and then the whole network for another 3-4 ??

## References from
- https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
- https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
- https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/84790
