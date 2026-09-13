# Skin Lesion Classification

This repository contains PyTorch scripts for training deep learning models to classify skin lesions. It is designed to work with the 7-class HAM10000 (ISIC) dataset. 

## Models Included

This project includes training pipelines for two powerful convolutional neural networks:
- **ResNet-50**: `model1_resnet_skin_cancer.py`
- **DenseNet-121**: `model2_densenet_skin_cancer.py`

Both models initialize with ImageNet pre-trained weights and have their final classification layers modified to predict 7 classes of skin diseases.

## Prerequisites

Make sure you have Python installed, along with the required libraries:
```bash
pip install torch torchvision
```

## Dataset Setup

The models expect the dataset to be structured in a standard PyTorch `ImageFolder` format. You will need to download the HAM10000 dataset and organize it into `train` and `val` directories within a `data` folder at the root of the project:

```text
data/
├── train/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
└── val/
    ├── akiec/
    ├── bcc/
    ├── ...
```

## Usage

Once your data is prepared, you can start training either model by running the respective script.

To train the ResNet-50 model:
```bash
python model1_resnet_skin_cancer.py
```

To train the DenseNet-121 model:
```bash
python model2_densenet_skin_cancer.py
```

## Features

- **Data Augmentation**: Includes RandomResizedCrop and RandomHorizontalFlip for training to prevent overfitting.
- **Normalization**: Standard ImageNet normalization is applied to all inputs.
- **Optimization**: Uses SGD with momentum and a StepLR learning rate scheduler for stable convergence.
