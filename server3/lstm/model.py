import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=126, hidden_size=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 128)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
