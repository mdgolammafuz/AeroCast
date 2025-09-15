# GRU model
import torch.nn as nn

class GRUWeatherForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super(GRUWeatherForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n.squeeze(0))
        return out
