import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from PIL import Image
from define import EEGDataset, EEGEncoder

# Model Loading with Fix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGEncoder().to(device)

best_dir = "checkpoint_epoch_4.pth"  # Change as needed
checkpoint = torch.load(best_dir, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
model.eval()
print("✅ Model loaded successfully.")

# Data Loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = EEGDataset('data.pth', 'imageset', transform=transform)

test_size = int(0.15 * len(dataset))
test_indices = list(range(len(dataset)))[-test_size:]
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluation
all_test_preds, all_test_labels, all_image_names = [], [], []

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

# Confusion Matrix
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
cm = confusion_matrix(all_test_labels, all_test_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Venustas', 'Firmitas', 'Utilitas'],
            yticklabels=['Venustas', 'Firmitas', 'Utilitas'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Set Confusion Matrix')
plt.savefig(f'{output_dir}/test_confusion_matrix.png', bbox_inches='tight')
plt.show()

# Save Predictions to CSV
results_df = pd.DataFrame({
    'Image_Name': all_image_names,
    'True_Label': [ ['Venustas', 'Firmitas', 'Utilitas'][label] for label in all_test_labels ],
    'Predicted_Label': [ ['Venustas', 'Firmitas', 'Utilitas'][pred] for pred in all_test_preds ],
    'Correct': [pred == true for pred, true in zip(all_test_preds, all_test_labels)]
})
results_df.to_csv(f'{output_dir}/eeg_predictions.csv', index=False)

print(f'Visualization saved in {output_dir}')
