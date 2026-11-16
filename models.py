import torch
import torch.nn as nn
import torch.nn.functional as F


class HAR_CNN_LSTM(nn.Module):
    """
    PyTorch execution module matching the Keras 3 Sequential architecture in cnn_lstm_model.keras:
    Input (B, 128, 9) -> Conv1D(64, k=3, ReLU) -> Conv1D(64, k=3, ReLU) -> Dropout(0.5)
    -> MaxPool1D(2) -> LSTM(100, return_sequences=False) -> Dropout(0.5)
    -> Dense(100, ReLU) -> Dense(6, Softmax/Logits).
    """

    def __init__(self, in_channels: int = 9, num_classes: int = 6):
        super(HAR_CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either (B, 128, 9) timesteps-first or (B, 9, 128) channels-first
        if x.ndim == 3 and x.shape[1] == 128 and x.shape[2] == 9:
            x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (B, 62, 64) for batch_first LSTM
        lstm_out, _ = self.lstm(x)
        x = self.dropout2(lstm_out[:, -1, :])
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
