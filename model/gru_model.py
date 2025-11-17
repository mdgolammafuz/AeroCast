import torch
import torch.nn as nn


class GRUWeatherForecaster(nn.Module):
    """
    Generic GRU forecaster:
      - input:  (batch, seq_len, input_dim)
      - output: (batch, output_dim)  (1-step forecast)

    Callers MUST pass the correct input_dim:
      - AeroCast pipeline (NOAA / sim): input_dim = 3
      - NOAA benchmark with time features: input_dim = 5
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Only apply dropout if we have stacked layers; PyTorch ignores otherwise,
        # but we keep it explicit.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, output_dim)
        """
        gru_out, _ = self.gru(x)          # (batch, seq_len, hidden_dim)
        last_hidden = gru_out[:, -1, :]   # (batch, hidden_dim)
        out = self.fc(last_hidden)        # (batch, output_dim)
        return out
