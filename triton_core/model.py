"""Neural-network primitives shared by active Triton Lite models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation, padding=self.padding
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding:
            y = y[..., :-self.padding]
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        return F.gelu(y + self.residual(x))


class CausalEventEncoder(nn.Module):
    """Encode forcing through the requested time without future leakage."""

    def __init__(self, input_features, hidden_channels, embedding_dim, dropout, layers):
        super().__init__()
        if layers < 1:
            raise ValueError("The causal event encoder requires at least one layer")
        channels = [input_features] + [hidden_channels] * (layers - 1) + [embedding_dim]
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(
                    channels[index], channels[index + 1], 2**index, dropout
                )
                for index in range(layers)
            ]
        )

    def forward(self, event: torch.Tensor, time_index: torch.Tensor) -> torch.Tensor:
        x = event.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        gather_index = time_index.view(-1, 1, 1).expand(-1, x.shape[1], 1)
        return x.gather(dim=2, index=gather_index).squeeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

