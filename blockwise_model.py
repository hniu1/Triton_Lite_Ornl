import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_features, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.squeeze(-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, output_dim: int, dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockwiseFloodModel(nn.Module):
    def __init__(
        self,
        event_features: int,
        block_features: int,
        temporal_channels: int = 64,
        event_embedding_dim: int = 64,
        block_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            input_features=event_features,
            hidden_channels=temporal_channels,
            embedding_dim=event_embedding_dim,
            dropout=dropout,
        )
        self.block_encoder = MLP(
            input_dim=block_features,
            hidden_dims=[block_hidden_dim],
            output_dim=block_hidden_dim,
            dropout=dropout,
        )
        self.predictor = MLP(
            input_dim=event_embedding_dim + block_hidden_dim,
            hidden_dims=[fusion_hidden_dim, fusion_hidden_dim // 2],
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, event_tensor: torch.Tensor, block_features: torch.Tensor) -> torch.Tensor:
        event_embedding = self.temporal_encoder(event_tensor)
        block_embedding = self.block_encoder(block_features)
        fused = torch.cat([event_embedding, block_embedding], dim=-1)
        return self.predictor(fused).squeeze(-1)
