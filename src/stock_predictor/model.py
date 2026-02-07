import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim_1=128, hidden_dim_2=64, output_dim=1):
        super(StockPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim_2, 25)
        self.fc2 = nn.Linear(25, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out