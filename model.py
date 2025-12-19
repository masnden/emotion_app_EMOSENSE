import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64 * 12 * 12,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, 1, 48, 48)
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.cnn(x)
        x = x.view(b, t, -1)

        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)
