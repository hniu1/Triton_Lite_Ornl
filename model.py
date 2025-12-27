import torch.nn as nn


import torch
import torch.nn as nn

class TritonCNN(nn.Module):
    """1D CNN model compatible with Keras Conv1D input layout (batch, steps, features),
    now including dropout for regularization."""

    def __init__(self,
                 in_features: int,
                 out_dim: int,
                 conv1_filters: int,
                 conv2_filters: int,
                 dense1_units: int,
                 dense2_units: int,
                 dense3_units: int,
                 dropout: float = 0.0):
        super(TritonCNN, self).__init__()

        # --- Convolutional feature extractor ---
        self.conv1 = nn.Conv1d(in_channels=in_features,
                               out_channels=conv1_filters,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=conv1_filters,
                               out_channels=conv2_filters,
                               kernel_size=1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(conv2_filters, dense1_units)
        self.fc2 = nn.Linear(dense1_units, dense2_units)
        self.fc3 = nn.Linear(dense2_units, dense3_units)
        self.out = nn.Linear(dense3_units, out_dim)

        # --- Dropout layer ---
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Input comes in as (batch, steps, features)
        x = x.permute(0, 2, 1)  # → (batch, features, steps)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        # Apply dropout between fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        return self.out(x)

