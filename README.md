# Human Activity Recognition with MLP (PyTorch)

This project implements a Multi-Layer Perceptron (MLP) in PyTorch to classify human activities from smartphone sensor data (UCI HAR Dataset). The model achieves 95.8% test accuracy after hyperparameter optimization with Optuna.

## Features

- PyTorch implementation of MLP with BatchNorm and Dropout
- Hyperparameter optimization using Optuna with 3-fold CV
- Complete reproducible pipeline (seeding for all random operations)
- Early stopping and model checkpointing
- Detailed evaluation metrics (confusion matrix, classification report)
- Training/validation loss visualization

## Dataset

UCI Human Activity Recognition (HAR) Dataset:
- 6 activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)
- 561 features from smartphone accelerometer and gyroscope data
- 7,352 training samples, 2,947 test samples

## Results

- Test Accuracy: 95.8%
- Best Hyperparameters:
  - Learning Rate: 2e-4
  - Weight Decay: 2e-7
  - Architecture: 2 hidden layers (128 units each)
  - Dropout Rates: 0.19 (layer 1), 0.14 (layer 2)

## Requirements

See [requirements.txt](requirements.txt)