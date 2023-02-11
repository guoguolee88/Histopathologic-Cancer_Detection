# Histopathologic-Cancer-Detection
The analysis is for a study of deep learning class. The purpose is to create an algorithm to identify Identify metastatic tissue in histopathologic scans of lymph node sections taken from larger digital pathology scans. The data for this analysis is a slightly modified version of the PatchCamelyon(PCam) benchmark dataset.

## Data
- PatchCamelyon (PCam)
Kaggle Competition: https://www.kaggle.com/c/histopathologic-cancer-detection

## Data Visualization
![image](https://user-images.githubusercontent.com/98787809/218280897-22f7661d-c851-490a-be95-a12ad019c840.png)


## Baseline Model
- 

## Baseline Model tuning
- 

## Validation and Analysis
- Data split applied
  - data class balancing
  - WSI (Whole slide imaging)
- Random augmentation from imgaug package: flips, rotations, crops, saturation
- Using tfrecord.
- Applied SENet (from https://github.com/kobiso/SENet-tensorflow-slim)
- Validation TTA - TenCrop

## References from
- https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
- https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
- https://www.kaggle.com/c/histopathologic-cancer-detection/discussion/84790
- https://github.com/kobiso/SENet-tensorflow-slim
