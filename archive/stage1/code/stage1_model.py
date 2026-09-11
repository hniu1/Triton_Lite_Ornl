"""Timestamp-conditioned spatial surrogate for TRITON trajectories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation, padding=padding
        )
        # LayerNorm is applied independently at each timestamp. BatchNorm1d
        # would aggregate statistics across the full temporal axis during
        # training and leak future forcing into earlier representations.
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
    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        embedding_dim: int,
        dropout: float,
        layers: int,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("The causal event encoder requires at least one layer")
        channels = [input_features] + [hidden_channels] * (layers - 1) + [embedding_dim]
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(channels[i], channels[i + 1], dilation=2**i, dropout=dropout)
                for i in range(layers)
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


class Stage1TimestampModel(nn.Module):
    def __init__(
        self,
        event_features: int,
        block_features: int,
        static_channels: int,
        temporal_channels: int = 96,
        temporal_layers: int = 8,
        event_embedding_dim: int = 128,
        conditioning_dim: int = 128,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.event_encoder = CausalEventEncoder(
            event_features,
            temporal_channels,
            event_embedding_dim,
            dropout,
            temporal_layers,
        )
        self.block_encoder = nn.Sequential(
            nn.Linear(block_features, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 64)
        )
        self.time_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32))
        self.conditioning = nn.Sequential(
            nn.Linear(event_embedding_dim + 64 + 32, conditioning_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.enc1 = ConvBlock(static_channels + 1, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4 + conditioning_dim, base_channels * 8)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.depth_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.wet_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.component_head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(
        self,
        event: torch.Tensor,
        time_index: torch.Tensor,
        time_features: torch.Tensor,
        block_features: torch.Tensor,
        static: torch.Tensor,
        mask: torch.Tensor,
        shared_event_time: bool = False,
    ):
        if shared_event_time:
            event_embedding = self.event_encoder(event[:1], time_index[:1]).expand(
                event.shape[0], -1
            )
        else:
            event_embedding = self.event_encoder(event, time_index)
        block_embedding = self.block_encoder(block_features)
        time_embedding = self.time_encoder(time_features)
        condition = self.conditioning(
            torch.cat([event_embedding, block_embedding, time_embedding], dim=1)
        )

        x1 = self.enc1(torch.cat([static, mask.unsqueeze(1)], dim=1))
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))
        pooled = F.avg_pool2d(x3, 2)
        condition_map = condition[:, :, None, None].expand(
            -1, -1, pooled.shape[2], pooled.shape[3]
        )
        x = self.bottleneck(torch.cat([pooled, condition_map], dim=1))
        x = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, x1], dim=1))

        depth = F.softplus(self.depth_head(x)).squeeze(1) * mask
        wet_logits = self.wet_head(x).squeeze(1)
        components = self.component_head(x) * mask.unsqueeze(1)
        return depth, wet_logits, components[:, 0], components[:, 1]
