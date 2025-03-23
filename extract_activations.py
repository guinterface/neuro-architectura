import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import mne
from torch.utils.data import DataLoader, Subset
from define import EEGDataset, EEGEncoder  # Import dataset and model

# ---------------------------
# 🔹 Load Model and Data
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = EEGDataset('data.pth', 'imageset')

# Use only the test portion (15% of the dataset)
test_size = int(0.15 * len(dataset))
test_indices = list(range(len(dataset))[-test_size:])
test_dataset = Subset(dataset, test_indices)

# Custom DataLoader (excluding images)
def custom_collate(batch):
    eeg_tensors = torch.stack([item[0] for item in batch])  # EEG signals
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)  # Labels
    return eeg_tensors, labels

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

# Load trained model
checkpoint_path = 'checkpoint_epoch_4.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model = EEGEncoder().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded model from {checkpoint_path}, trained for {checkpoint['epoch']} epochs.")

# ---------------------------
# 🔹 Extract Spatial Activations
# ---------------------------
def extract_spatial_activations(model, dataloader):
    """
    Extracts spatial EEG activations by averaging across temporal filters.
    """
    model.eval()
    activations_dict = {'Venustas': [], 'Firmitas': [], 'Utilitas': []}

    with torch.no_grad():
        for eeg, labels in dataloader:
            eeg = eeg.to(device)
            _ = model(eeg)  # Forward pass
            
            # Retrieve activations from temporal_out
            if 'temporal_out' not in model.activations:
                raise ValueError("❌ Model is not storing 'temporal_out'. Check the EEGEncoder forward method!")

            activation = model.activations['temporal_out']  # Shape (batch, 50, 128, X)
            activation = activation.mean(dim=1).cpu().numpy()  # ✅ Average over 50 filters → (batch, 128, X)
            print(f"🔹 Extracted shape after averaging: {activation.shape}")  # Debugging print
            
            for i, label in enumerate(labels.numpy()):
                class_name = ['Venustas', 'Firmitas', 'Utilitas'][label]
                activations_dict[class_name].append(activation[i])

    return activations_dict



# Extract activations
activations = extract_spatial_activations(model, test_loader)

# ---------------------------
# 🔹 Compute & Save Mean Activations
# ---------------------------
def compute_mean_activations(activations):
    return {class_name: np.mean(np.stack(acts), axis=0) for class_name, acts in activations.items() if acts}

mean_activations = compute_mean_activations(activations)

# ---------------------------
# 🔹 Generate Brain Activation Maps
# ---------------------------
# Define standard 128-channel EEG montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')

def plot_brain_activation_map(activation_values, class_name):
    """
    Generates and saves a topographic brain activation map.
    """
    num_channels = 128  # EEG dataset uses 128 channels
    print(f"🔹 Shape {activation_values.shape}")  # Debugging print

    # Ensure activation values are in (128, X) shape
    if activation_values.shape[0] != num_channels:
        print(f"⚠️ Warning: Expected 128 channels, but got {activation_values.shape[0]}")
        return

    # Average over the X dimension (e.g., temporal or spatial)
    activation_values = activation_values.mean(axis=-1)  # Average across last axis (time)
    if activation_values.shape[0] != 128:
        raise ValueError(f"❌ Activation shape mismatch: Expected 128, got {activation_values.shape}")

    # Create an MNE info structure
    info = mne.create_info(ch_names=montage.ch_names, sfreq=1000, ch_types='eeg')
    info.set_montage(montage)

    # Convert activation values to MNE's EvokedArray format
    evoked = mne.EvokedArray(activation_values[:, np.newaxis], info)

    # Generate the topographic plot
    fig, ax = plt.subplots(figsize=(8, 6))
    mne.viz.plot_topomap(
        evoked.data[:, 0], evoked.info, axes=ax, show=False, extrapolate='local', contours=0, outlines=None
    )
    # Save the figure
    output_path = f"output/brain_activation_{class_name}.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"📁 Brain activation map for {class_name} saved: {output_path}")

# ---------------------------
# 🔹 Process Each Class Activation
# ---------------------------
for class_name, activation in mean_activations.items():
    plot_brain_activation_map(activation, class_name)

print("✅ EEG Brain Activation Maps Generated Successfully!")








          