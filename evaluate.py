import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import model structures
from model1_resnet_skin_cancer import get_resnet50_model
from model2_densenet_skin_cancer import get_densenet_model

def evaluate_model(model, dataloader, device, class_names, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = f'confusion_matrix_{model_name}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot as {save_path}\n")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # Data transformation for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists('data/val'):
        print("Error: data/val directory not found.")
        exit()

    val_dataset = datasets.ImageFolder('data/val', val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = val_dataset.classes

    num_classes = len(class_names)

    # Evaluate ResNet50
    if os.path.exists('resnet50_skin_cancer.pth'):
        resnet = get_resnet50_model(num_classes).to(device)
        resnet.load_state_dict(torch.load('resnet50_skin_cancer.pth', map_location=device))
        evaluate_model(resnet, val_loader, device, class_names, "ResNet50")
    else:
        print("resnet50_skin_cancer.pth not found. Skipping ResNet50 evaluation.")

    # Evaluate DenseNet121
    if os.path.exists('densenet121_skin_cancer.pth'):
        densenet = get_densenet_model(num_classes).to(device)
        densenet.load_state_dict(torch.load('densenet121_skin_cancer.pth', map_location=device))
        evaluate_model(densenet, val_loader, device, class_names, "DenseNet121")
    else:
        print("densenet121_skin_cancer.pth not found. Skipping DenseNet121 evaluation.")
