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

## Applied
- data class balancing

## TODO 
- data analysis.
- data transformation. (center crop, etc...)
- reference with kaggle kernel
- apply other networks (Inception V4, Inception-ResNet-v2, and ?)
- multiple crops at multiple scales.
- make ensemble module.

## References from
- https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
- https://github.com/balancap/SSD-Tensorflow
- https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
