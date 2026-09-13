# Research Paper Progress Update

## Accomplished Tasks
- Successfully set up the initial PyTorch models (`ResNet-50` and `DenseNet-121`).
- Resolved missing dependency issues (installed `torch` and `torchvision`).
- Fully completed the training loops in both scripts (`model1_resnet_skin_cancer.py` and `model2_densenet_skin_cancer.py`).
- Integrated DataLoaders with essential data augmentations (RandomResizedCrop, HorizontalFlip) and normalizations.
- Added a full training and validation loop using CrossEntropyLoss and SGD with StepLR scheduler.
- Scripts now expect the dataset to be placed in a `data/` folder with `train` and `val` splits.

## Next Steps for the Team
- **Dataset Acquisition**: We need to download the full `HAM10000` / `ISIC` dataset.
- **Data Structuring**: The dataset should be arranged in the standard `ImageFolder` structure:
  ```
  data/
  ├── train/
  │   ├── class1/
  │   ├── class2/
  │   └── ...
  └── val/
      ├── class1/
      ├── class2/
      └── ...
  ```
- **Training Runs**: Once the data is in place, the scripts can be executed directly to train the models and get evaluation metrics (loss and accuracy).

*Prepared for the Wednesday Weekly Research Meeting.*
