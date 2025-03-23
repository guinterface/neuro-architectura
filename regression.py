import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

# ==========================
# 1. LOAD RESNET MODEL
# ==========================

# Load pretrained ResNet
resnet = models.resnet50(pretrained=True)
resnet.eval()  # Set to evaluation mode

# Select layers for feature extraction
selected_layers = {
    "conv1": resnet.conv1,
    "layer1": resnet.layer1,
    "layer2": resnet.layer2,
    "layer3": resnet.layer3,
    "layer4": resnet.layer4,
    "avgpool": resnet.avgpool,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# ==========================
# 2. IMAGE FEATURE EXTRACTION
# ==========================

# Dataset path
DATA_DIR = "/home/guisi/final/imageset"  # Update with actual dataset path

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(root=DATA_DIR, transform=transform)

# Extract features from multiple layers
features_dict = {layer: [] for layer in selected_layers}
image_names = []

# Hook function to capture layer activations
activations = {}

def hook_fn(layer_name):
    def hook(module, input, output):
        activations[layer_name] = output.detach()
    return hook

# Register hooks
hooks = []
for layer_name, layer in selected_layers.items():
    hook = layer.register_forward_hook(hook_fn(layer_name))
    hooks.append(hook)

# Process all images
for image_path, _ in dataset.imgs:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    activations.clear()  # Reset activations before forward pass
    with torch.no_grad():
        _ = resnet(image)

    # Store activations from each selected layer
    for layer_name in selected_layers:
        features = activations[layer_name].flatten().cpu().numpy()
        features_dict[layer_name].append(features)

    image_names.append(os.path.basename(image_path))

# Convert to NumPy arrays and save
for layer_name, features in features_dict.items():
    np.save(f"resnet_features_{layer_name}.npy", np.array(features))

np.save("image_names.npy", np.array(image_names))

# Remove hooks
for hook in hooks:
    hook.remove()

print("ResNet features extracted for all layers.")

# ==========================
# 3. EEG FEATURE EXTRACTION
# ==========================

class EEGDataset:
    def __init__(self, pth_file, image_root):
        self.data = torch.load(pth_file)
        self.image_root = image_root
        self.label_map = {"venustas": 0, "firmitas": 1, "utilitas": 2}
    
    def __len__(self):
        return len(self.data['dataset'])
    
    def __getitem__(self, idx):
        eeg = self.data['dataset'][idx]['eeg']  # Shape: (1, 128, 250)
        eeg = (eeg - eeg.mean()) / eeg.std()
        label = self.data['dataset'][idx]['label'].lower().split('/')[0]
        label = self.label_map.get(label, -1)
        image_name = self.data['image'][idx]
        return eeg, label, image_name

# Load EEG dataset
EEG_DATA_PATH = "/home/guisi/final/data.pth"
IMAGE_ROOT = "/home/guisi/final/imageset"
eeg_dataset = EEGDataset(EEG_DATA_PATH, IMAGE_ROOT)

eeg_features = []
eeg_labels = []
eeg_image_names = []

for idx in range(len(eeg_dataset)):
    eeg, label, image_name = eeg_dataset[idx]

    # Extract EEG feature vector (mean across time)
    eeg_feature_vector = eeg.mean(dim=-1).flatten().numpy()

    eeg_features.append(eeg_feature_vector)
    eeg_labels.append(label)
    eeg_image_names.append(image_name)

# Save EEG features
np.save("eeg_features.npy", np.array(eeg_features))
np.save("eeg_labels.npy", np.array(eeg_labels))
np.save("eeg_image_names.npy", np.array(eeg_image_names))

print("EEG features extracted and saved.")

# ==========================
# 4. MAP IMAGE FEATURES TO EEG RESPONSES
# ==========================

# Load EEG data
Y = np.load("eeg_features.npy")  # EEG responses
image_names_eeg = np.load("eeg_image_names.npy")

# Ridge regression results per layer
scores = {}

for layer_name in selected_layers:
    print(f"Processing layer: {layer_name}")

    # Load ResNet features for this layer
    X = np.load(f"resnet_features_{layer_name}.npy")  # Image features
    image_names_resnet = np.load("image_names.npy")

    # Align EEG and image features by matching filenames
    matched_X, matched_Y = [], []

    for i, img_name in enumerate(image_names_eeg):
        if img_name in image_names_resnet:
            idx = list(image_names_resnet).index(img_name)
            matched_X.append(X[idx])
            matched_Y.append(Y[i])

    matched_X = np.array(matched_X)
    matched_Y = np.array(matched_Y)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(matched_X, matched_Y, test_size=0.2, random_state=42)

    # Train Ridge Regression
    ridge = RidgeCV(alphas=[1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100])
    ridge.fit(X_train, Y_train)

    # Predict EEG responses
    Y_pred = ridge.predict(X_test)

    # Compute R² score
    r2 = r2_score(Y_test, Y_pred, multioutput="raw_values")
    mean_r2 = np.mean(r2)

    print(f"Layer: {layer_name}, Mean R² Score: {mean_r2:.3f}")

    # Store score for heatmap
    scores[layer_name] = mean_r2

# ==========================
# 5. VISUALIZATION
# ==========================

# Convert scores to DataFrame
df_scores = pd.DataFrame(scores, index=["R²"]).T

# Create heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df_scores, annot=True, cmap="viridis", fmt=".3f")
plt.title("EEG Predictability (R²) Across ResNet Layers")
plt.xlabel("Metric")
plt.ylabel("ResNet Layer")
plt.tight_layout()
plt.savefig("heatmap_resnet_eeg.png", dpi=300)
plt.show()

print("Heatmap saved as heatmap_resnet_eeg.png")
