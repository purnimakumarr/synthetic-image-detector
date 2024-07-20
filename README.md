# Image Classification: Real vs Fake (AI Generated Synthetic)

This repository contains the code and resources for the image classification project that distinguishes between real and AI-generated synthetic images using the CIFAKE dataset. The project utilizes TensorFlow, Keras, and Keras Tuner for hyperparameter optimization.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Introduction
The goal of this project is to build a robust image classification model to distinguish between real and AI-generated (synthetic) images. The model leverages convolutional neural networks (CNNs) and hyperparameter tuning to achieve high accuracy.

## Dataset
The dataset used in this project is a subset of the CIFAKE dataset, which contains both real and AI-generated synthetic images. The images are 32x32 pixels in size.

- Training images: 80,000
- Validation images: 20,000
- Test images: 20,000

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Keras Tuner
- Matplotlib
- NumPy

You can install the necessary packages using the following command:
```bash
pip install tensorflow keras keras-tuner matplotlib numpy
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/purnimakumarr/image-classification-cifake.git
cd image-classification-cifake
```

2. Load the dataset and train the model:
```bash
python src/train.py
```

## Model Training
The model is defined and trained using TensorFlow and Keras. The `Hyperband` tuner from Keras Tuner is used for hyperparameter optimization. The key components of the model include:
- Rescaling
- Convolutional layers with ReLU or Tanh activation
- Max Pooling
- Dense layers with ReLU or Tanh activation
- Dropout for regularization
- Early stopping and model checkpoints

## Evaluation
The model is evaluated using accuracy, precision, recall, and F1 score. The training history is plotted to visualize the performance metrics over epochs.

## Results
The optimal hyperparameters found are:
- Number of Conv2D layers: 2
- Conv2D Layer 1: 256 filters, 3x3 kernel size, ReLU activation
- Conv2D Layer 2: 32 filters, 3x3 kernel size, ReLU activation
- Number of Dense layers: 1
- Dense Layer 1: 256 units, ReLU activation
- Optimizer: Adam
- Learning rate: 0.000147

The training and validation accuracy, loss, precision, and recall are plotted and saved in the `results` directory.

## License
This project is licensed under the MIT License.

## Authors
Purnima Kumar
Department of Computer Science
University of Delhi