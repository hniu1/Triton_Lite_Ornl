from typing import Optional

import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class StaticRasterEncoder(nn.Module):
    """Lightweight CNN that compresses C_static × 80 × 80 static hydraulic channels
    into a spatial feature map of shape raster_enc_channels × 80 × 80."""

    def __init__(self, in_channels: int, out_channels: int = 16) -> None:
        super().__init__()
        mid = max(out_channels, in_channels * 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockwiseFloodMatrixModel(nn.Module):
    def __init__(
        self,
        event_features: int,
        block_features: int,
        target_rows: int,
        target_cols: int,
        temporal_channels: int = 64,
        event_embedding_dim: int = 64,
        block_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        decoder_base_channels: int = 128,
        dropout: float = 0.1,
        static_raster_channels: int = 0,
        raster_enc_channels: int = 16,
        predict_velocity: bool = False,
        predict_velocity_magnitude: bool = False,
    ) -> None:
        super().__init__()
        if target_rows != 80 or target_cols != 80:
            raise ValueError(
                f"BlockwiseFloodMatrixModel currently expects 80x80 outputs, got {target_rows}x{target_cols}"
            )

        self.target_rows = target_rows
        self.target_cols = target_cols
        self.predict_velocity = predict_velocity
        self.predict_velocity_magnitude = predict_velocity_magnitude
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
        self.fusion = MLP(
            input_dim=event_embedding_dim + block_hidden_dim,
            hidden_dims=[fusion_hidden_dim],
            output_dim=decoder_base_channels * 10 * 10,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            UpsampleBlock(decoder_base_channels, decoder_base_channels // 2),
            UpsampleBlock(decoder_base_channels // 2, decoder_base_channels // 4),
            UpsampleBlock(decoder_base_channels // 4, decoder_base_channels // 8),
        )

        # Optional static raster encoder (channels = 0 means disabled; backwards-compatible)
        self.static_raster_channels = static_raster_channels
        self.raster_enc_channels = raster_enc_channels if static_raster_channels > 0 else 0
        if static_raster_channels > 0:
            self.raster_encoder = StaticRasterEncoder(
                in_channels=static_raster_channels,
                out_channels=raster_enc_channels,
            )
        else:
            self.raster_encoder = None  # type: ignore[assignment]

        head_channels = (decoder_base_channels // 8) + 3 + self.raster_enc_channels
        self.depth_head = nn.Sequential(
            nn.Conv2d(head_channels, decoder_base_channels // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_base_channels // 8, 1, kernel_size=1),
        )
        self.wet_head = nn.Sequential(
            nn.Conv2d(head_channels, decoder_base_channels // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_base_channels // 8, 1, kernel_size=1),
        )
        if self.predict_velocity:
            self.velocity_x_head = nn.Sequential(
                nn.Conv2d(head_channels, decoder_base_channels // 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(decoder_base_channels // 8, 1, kernel_size=1),
            )
            self.velocity_y_head = nn.Sequential(
                nn.Conv2d(head_channels, decoder_base_channels // 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(decoder_base_channels // 8, 1, kernel_size=1),
            )
        else:
            self.velocity_x_head = None  # type: ignore[assignment]
            self.velocity_y_head = None  # type: ignore[assignment]

        if self.predict_velocity_magnitude:
            self.velocity_magnitude_head = nn.Sequential(
                nn.Conv2d(head_channels, decoder_base_channels // 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(decoder_base_channels // 8, 1, kernel_size=1),
            )
        else:
            self.velocity_magnitude_head = None  # type: ignore[assignment]
        self.output_activation = nn.Softplus()

    def _coordinate_channels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        y_coords = torch.linspace(-1.0, 1.0, self.target_rows, device=device)
        x_coords = torch.linspace(-1.0, 1.0, self.target_cols, device=device)
        y_grid = y_coords.view(1, 1, self.target_rows, 1).expand(batch_size, 1, self.target_rows, self.target_cols)
        x_grid = x_coords.view(1, 1, 1, self.target_cols).expand(batch_size, 1, self.target_rows, self.target_cols)
        return torch.cat([y_grid, x_grid], dim=1)

    def forward(
        self,
        event_tensor: torch.Tensor,
        block_features: torch.Tensor,
        block_mask: torch.Tensor,
        static_raster: Optional[torch.Tensor] = None,
    ) -> tuple:
        event_embedding = self.temporal_encoder(event_tensor)
        block_embedding = self.block_encoder(block_features)
        fused = torch.cat([event_embedding, block_embedding], dim=-1)

        decoded = self.fusion(fused).view(fused.shape[0], -1, 10, 10)
        decoded = self.decoder(decoded)

        mask_channel = block_mask.unsqueeze(1)
        coord_channels = self._coordinate_channels(batch_size=fused.shape[0], device=fused.device)
        parts = [decoded, mask_channel, coord_channels]
        if self.raster_encoder is not None:
            if static_raster is None:
                # Fallback: substitute zero-filled raster (e.g., when running inference
                # without providing the optional static raster dir)
                static_raster = torch.zeros(
                    decoded.shape[0], self.static_raster_channels, self.target_rows, self.target_cols,
                    device=decoded.device, dtype=decoded.dtype,
                )
            parts.append(self.raster_encoder(static_raster))
        fused_map = torch.cat(parts, dim=1)
        depth_logits = self.depth_head(fused_map)
        wet_logits = self.wet_head(fused_map).squeeze(1)
        depth_map = self.output_activation(depth_logits).squeeze(1) * block_mask

        outputs = [depth_map, wet_logits]
        if self.predict_velocity and self.velocity_x_head is not None and self.velocity_y_head is not None:
            velocity_x_map = self.velocity_x_head(fused_map).squeeze(1) * block_mask
            velocity_y_map = self.velocity_y_head(fused_map).squeeze(1) * block_mask
            outputs.extend([velocity_x_map, velocity_y_map])

        if self.predict_velocity_magnitude and self.velocity_magnitude_head is not None:
            velocity_magnitude_map = self.output_activation(self.velocity_magnitude_head(fused_map)).squeeze(1) * block_mask
            outputs.append(velocity_magnitude_map)

        return tuple(outputs)
