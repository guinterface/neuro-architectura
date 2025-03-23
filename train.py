import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Subset
import random 
from define import EEGDataset, EEGEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

from PIL import Image


torch.cuda.empty_cache()
# EEG Dataset

# Hyperparameters
epochs = 10
batch_size = 16
learning_rate = 0.001
betas = (0.9, 0.999)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = EEGDataset('data.pth', 'imageset', transform=transform)

# Split dataset (e.g., 70% train, 15% validation, 15% test)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Generate global indices for the full dataset
full_indices = list(range(len(full_dataset)))

random.shuffle(full_indices)
train_indices = full_indices[:train_size]
val_indices = full_indices[train_size:train_size + val_size]
test_indices = full_indices[train_size + val_size:]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

indices_df = pd.DataFrame({
    'index': full_indices,
    'split': ['train'] * len(train_indices) + ['val'] * len(val_indices) + ['test'] * len(test_indices)
})
indices_df.to_csv("dataset_splits.csv", index=False)


# Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset Split: Train({len(train_dataset)}), Validation({len(val_dataset)}), Test({len(test_dataset)})")

# Model, Loss, and Optimizer
model = EEGEncoder().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

# Training Loop

best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (eeg, _, labels, _) in enumerate(train_loader):
        eeg, labels = eeg.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}')

    train_accuracy = 100. * correct / total
    train_loss = running_loss / len(train_loader)

    # 🔹 **Validation Step**
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for eeg, _, labels, _ in val_loader:
            eeg, labels = eeg.to(device), labels.to(device)
            outputs = model(eeg)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100. * val_correct / val_total

    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
    

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'activations': model.activations  # Save activations
        }, f'checkpoint_epoch_{epoch + 1}.pth')
        print("✅ Best model saved.")
        best_dir = f'checkpoint_epoch_{epoch + 1}.pth'
model.eval()
all_preds, all_labels, all_image_names = [], [], []
model.load_state_dict(torch.load( best_dir))  # Load best model
model.eval()

all_test_preds, all_test_labels = [], []

with torch.no_grad():
    for eeg, _, labels, image_names in test_loader:
        eeg = eeg.to(device)
        outputs = model(eeg)
        _, preds = torch.max(outputs, 1)

        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.numpy())
        all_image_names.extend(image_names)

# Compute Test Accuracy
test_accuracy = accuracy_score(all_test_labels, all_test_preds) * 100
print(f"\n🔹 Final Test Accuracy: {test_accuracy:.2f}%")
