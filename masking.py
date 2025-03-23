import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from torch.utils.data import DataLoader, Subset
from define import EEGDataset, EEGEncoder
from torchvision import transforms

# ------------------ Model Loading ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGEncoder().to(device)

best_dir = "checkpoint_epoch_4.pth"  # Change as needed
checkpoint = torch.load(best_dir, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Model loaded successfully.")

# ------------------ Data Loading ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = EEGDataset('data.pth', 'imageset', transform=transform)

test_size = int(0.15 * len(dataset))
test_indices = list(range(len(dataset)))[-test_size:]
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ------------------ Channel Masking ------------------
num_channels = 128  # Adjust based on EEG setup
channel_importance = np.zeros(num_channels)

print("🔹 Performing EEG Channel Masking...")

start_time = time.time()

for channel in range(num_channels):
    total_correct = 0
    total_samples = 0

    for eeg, _, labels, _ in test_loader:
        eeg, labels = eeg.to(device), labels.to(device)

        # Mask out one EEG channel
        eeg_masked = eeg.clone()
        eeg_masked[:, :, channel, :] = 0  # Zero out this channel

        outputs = model(eeg_masked)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    accuracy_drop = 100 - (100 * total_correct / total_samples)
    channel_importance[channel] = accuracy_drop

    if channel % 10 == 0 or channel == num_channels - 1:  # Print every 10 channels
        elapsed = time.time() - start_time
        print(f"✅ Processed {channel+1}/{num_channels} channels. Time elapsed: {elapsed:.2f}s")

# Save and Plot Channel Importance
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


import pandas as pd

# Save Channel Importance to CSV
channel_importance_df = pd.DataFrame({
    "Channel": np.arange(num_channels),
    "Accuracy Drop (%)": channel_importance
})

csv_path = os.path.join(output_dir, "channel_importance.csv")
channel_importance_df.to_csv(csv_path, index=False)

print(f"📁 EEG Channel Importance saved to {csv_path}")

plt.figure(figsize=(10, 5))
plt.bar(range(num_channels), channel_importance)
plt.xlabel("EEG Channel")
plt.ylabel("Accuracy Drop (%)")
plt.title("EEG Channel Importance (Higher = More Important)")
plt.savefig(f"{output_dir}/channel_importance.png")
plt.show()


# ------------------ Final Summary ------------------
total_time = time.time() - start_time
print(f"\n✅ Masking analysis complete! Total execution time: {total_time / 60:.2f} minutes.")
print(f"📂 Results saved in {output_dir}.")

