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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

from PIL import Image

class EEGDataset(Dataset):
    def __init__(self, pth_file, image_root, transform=None):
        self.data = torch.load(pth_file)
        self.image_root = image_root
        self.transform = transform
        self.label_map = {"venustas": 0, "firmitas": 1, "utilitas": 2}
    
    def __len__(self):
        return len(self.data['dataset'])
    
    def __getitem__(self, idx):
        eeg = self.data['dataset'][idx]['eeg']  # Shape: (1, 128, 250)
        eeg = (eeg - eeg.mean()) / eeg.std()
        label = self.data['dataset'][idx]['label'].lower()
        
        # Clean the label in case there is extra folder structure like 'firmitas/firmitas'
        label = label.split('/')[0]
        
        # Check if the label exists in the map, if not print a warning
        if label not in self.label_map:
            print(f"Warning: Label '{label}' not in label_map.")
        
        label = self.label_map.get(label, -1)  # Default to -1 if label is missing
        
        image_name = self.data['image'][idx]  # Filename
        
        # Construct image path
        label_folder = image_name.split('_')[0]
        image_path = os.path.join(self.image_root, label_folder, label_folder.capitalize(), image_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return eeg, image, label, image_name


# EEG Model with Layer Capture
class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim=1000, num_classes=3):
        super(EEGEncoder, self).__init__()

        self.temporal_convs = nn.ModuleList([
            nn.Conv2d(1, 10, (1, 33), stride=(1, 2), dilation=(1, d), padding=(0, p))
            for d, p in zip([1, 2, 4, 8, 16], [16, 31, 62, 124, 248])
        ])

        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(1, 50, (k, 1), stride=(2, 1), padding=(p, 0))
            for k, p in zip([128, 64, 32, 16], [63, 31, 15, 7])
        ])

        self.residual_layers = nn.Sequential(
            ResidualLayer(250, (3, 3)),
            ResidualLayer(250, (3, 3)),
            ResidualLayer(250, (3, 3)),
            ResidualLayer(250, (3, 3))
        )

        self.final_conv = nn.Conv2d(250, 50, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        # Dynamically infer fc input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 250)
            
            temp_outs = [conv(dummy_input) for conv in self.temporal_convs]
            min_width = min([out.shape[-1] for out in temp_outs])
            temp_outs = [out[..., :min_width] for out in temp_outs]
            temporal_out = torch.cat(temp_outs, dim=1)

            spat_outs = [conv(dummy_input) for conv in self.spatial_convs]
            min_height = min([out.shape[-2] for out in spat_outs])
            spat_outs = [out[..., :min_height, :] for out in spat_outs]
            spatial_out = torch.cat(spat_outs, dim=1)

            min_height = min(temporal_out.shape[-2], spatial_out.shape[-2])
            min_width = min(temporal_out.shape[-1], spatial_out.shape[-1])
            temporal_out = temporal_out[..., :min_height, :min_width]
            spatial_out = spatial_out[..., :min_height, :min_width]

            x = torch.cat([temporal_out, spatial_out], dim=1)
            x = self.residual_layers(x)
            x = self.final_conv(x)

            fc_input_dim = x.view(1, -1).shape[1]  # correct size for fc input

        self.fc = nn.Linear(fc_input_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.activations = {}

    def forward(self, x):

        # Temporal Convolutions
        temp_outs = []
        for i, conv in enumerate(self.temporal_convs):
            out = conv(x)
            temp_outs.append(out)
        
        min_width = min([out.shape[-1] for out in temp_outs])
        temp_outs = [out[..., :min_width] for out in temp_outs]
        temporal_out = torch.cat(temp_outs, dim=1)
        self.activations['temporal_out'] = temporal_out.clone().detach()

        # Spatial Convolutions
        spat_outs = []
        for i, conv in enumerate(self.spatial_convs):
            out = conv(x)
            spat_outs.append(out)

        min_height = min([out.shape[-2] for out in spat_outs])
        spat_outs = [out[..., :min_height, :] for out in spat_outs]
        spatial_out = torch.cat(spat_outs, dim=1)
        self.activations['spatial_out'] = spatial_out.clone().detach()

        # Align temporal and spatial outputs
        min_height = min(temporal_out.shape[-2], spatial_out.shape[-2])
        min_width = min(temporal_out.shape[-1], spatial_out.shape[-1])
        temporal_out = temporal_out[..., :min_height, :min_width]
        spatial_out = spatial_out[..., :min_height, :min_width]

        x = torch.cat([temporal_out, spatial_out], dim=1)
        self.activations['temporal_spatial_concat'] = x.clone().detach()

        # Residual Layers
        for i, layer in enumerate(self.residual_layers):
            x = layer(x)
            self.activations[f'residual_layer_{i}'] = x.clone().detach()

        self.activations['residual_out'] = x.clone().detach()

        # Final Convolution
        x = self.final_conv(x)
        self.activations['final_conv_out'] = x.clone().detach()

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layer
        x = self.fc(x)
        self.activations['fc_out'] = x.clone().detach()

        # Classifier
        x = self.classifier(x)
        self.activations['classifier_out'] = x.clone().detach()

        return x



class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
