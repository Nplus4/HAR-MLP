# Human Activity Recognition using Deep Learning

This project uses the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) to classify human activities from smartphone sensor data using a PyTorch-based multi-layer perceptron (MLP).

## Features
- Preprocessing: Standardization, Label Encoding
- Model: 3-layer MLP with Batch Normalization, Dropout, ReLU activation
- Training: Early stopping based on validation loss
- Evaluation: Confusion matrix, classification report, test accuracy
- Visualization: Training/Validation loss plot, Confusion matrix heatmap

## Dataset
- The UCI Human Activity Recognition dataset contains accelerometer and gyroscope readings from 30 subjects performing 6 activities.

## Requirements
```bash
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
