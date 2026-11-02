import torch
import torch.nn as nn

class HAR_1DCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=6):
        super(HAR_1DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64

            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 64 -> 32

            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 32 -> 16
            nn.Dropout(p=0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        flattened = features.flatten(start_dim=1)
        logits = self.classifier(flattened)
        return logits


class RegularizedHAR_ANN(nn.Module):
    def __init__(self, input_dim=561, num_classes=6):
        super(RegularizedHAR_ANN, self).__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Output Layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)
