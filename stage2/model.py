"""Lead-time-conditioned direct hydraulic forecast model for Stage 2 V5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_core.model import CausalEventEncoder, ConvBlock


class Stage2DirectLeadModel(nn.Module):
    """Predict the state at ``origin + lead`` in one model application.

    The output is parameterized as a residual from the observed origin state,
    but it is a direct multi-horizon forecast: no intermediate prediction is
    recursively fed back into the network.
    """

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
        history_states: int = 2,
        history_fusion: str = "adapter",
        use_activity_gate: bool = True,
        activity_gate_initial_bias: float = -1.5,
        lead_feature_dim: int = 4,
        hydraulic_halo_cells: int = 0,
        local_forcing_embedding_dim: int = 0,
        use_component_correction_gate: bool = False,
        component_gate_initial_bias: float = -3.0,
    ) -> None:
        super().__init__()
        if history_states not in (1, 2):
            raise ValueError("history_states must be one or two")
        if history_fusion not in ("concat", "adapter"):
            raise ValueError("history_fusion must be 'concat' or 'adapter'")
        if history_states == 1 and history_fusion != "concat":
            raise ValueError("history_fusion='adapter' requires two history states")
        self.history_states = int(history_states)
        self.history_fusion = str(history_fusion)
        self.use_activity_gate = bool(use_activity_gate)
        self.hydraulic_halo_cells = int(hydraulic_halo_cells)
        self.local_forcing_embedding_dim = int(local_forcing_embedding_dim)
        self.use_component_correction_gate = bool(use_component_correction_gate)

        self.event_encoder = CausalEventEncoder(
            event_features,
            temporal_channels,
            event_embedding_dim,
            dropout,
            temporal_layers,
        )
        self.local_event_encoder = (
            CausalEventEncoder(
                1,
                max(16, temporal_channels // 2),
                self.local_forcing_embedding_dim,
                dropout,
                temporal_layers,
            )
            if self.local_forcing_embedding_dim > 0
            else None
        )
        self.block_encoder = nn.Sequential(
            nn.Linear(block_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.origin_time_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.lead_encoder = nn.Sequential(
            nn.Linear(lead_feature_dim, 32), nn.GELU(), nn.Linear(32, 32)
        )
        # Keep the V4b condition fields first so its learned condition matrix
        # can initialize the matching prefix exactly.
        condition_inputs = (
            event_embedding_dim
            + self.local_forcing_embedding_dim
            + 64
            + 32
            + 32
            + 32
        )
        self.conditioning = nn.Sequential(
            nn.Linear(condition_inputs, conditioning_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.enc1 = ConvBlock(static_channels + 1, base_channels)
        state_channels = (
            6
            if self.history_states == 2 and self.history_fusion == "concat"
            else 3
        )
        self.state_encoder = ConvBlock(state_channels, base_channels)
        self.halo_encoder = (
            ConvBlock(6 if history_states == 2 else 3, base_channels)
            if self.hydraulic_halo_cells > 0
            else None
        )
        self.history_adapter = (
            nn.Conv2d(3, 3, kernel_size=1)
            if self.history_states == 2 and self.history_fusion == "adapter"
            else None
        )
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(
            base_channels * 4 + conditioning_dim, base_channels * 8
        )
        self.dec3 = ConvBlock(
            base_channels * 8 + base_channels * 4, base_channels * 4
        )
        self.dec2 = ConvBlock(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.depth_delta_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.component_delta_head = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.wet_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.activity_head = (
            nn.Conv2d(base_channels, 1, kernel_size=1)
            if self.use_activity_gate
            else None
        )
        self.component_gate_head = (
            nn.Conv2d(base_channels, 1, kernel_size=1)
            if self.use_component_correction_gate
            else None
        )
        nn.init.zeros_(self.depth_delta_head.weight)
        nn.init.zeros_(self.depth_delta_head.bias)
        nn.init.zeros_(self.component_delta_head.weight)
        nn.init.zeros_(self.component_delta_head.bias)
        if self.activity_head is not None:
            nn.init.zeros_(self.activity_head.weight)
            nn.init.constant_(
                self.activity_head.bias, float(activity_gate_initial_bias)
            )
        if self.component_gate_head is not None:
            nn.init.zeros_(self.component_gate_head.weight)
            nn.init.constant_(
                self.component_gate_head.bias, float(component_gate_initial_bias)
            )
        if self.history_adapter is not None:
            nn.init.zeros_(self.history_adapter.weight)
            nn.init.zeros_(self.history_adapter.bias)

    def forward(
        self,
        event,
        target_time_index,
        target_time_features,
        origin_time_features,
        lead_features,
        block_features,
        static,
        mask,
        origin_depth,
        origin_component_x,
        origin_component_y,
        shared_event_time: bool = False,
        older_depth=None,
        older_component_x=None,
        older_component_y=None,
        forcing_weights=None,
        hydraulic_context=None,
        older_hydraulic_context=None,
    ):
        # Evaluating the causal encoder at target_time_index intentionally
        # admits forcing known through the requested valid time, never beyond.
        if shared_event_time:
            event_embedding = self.event_encoder(
                event[:1], target_time_index[:1]
            ).expand(event.shape[0], -1)
        else:
            event_embedding = self.event_encoder(event, target_time_index)
        condition_parts = [event_embedding]
        if self.local_event_encoder is not None:
            if forcing_weights is None:
                raise ValueError("Localized-forcing model requires forcing weights")
            local_event = (event * forcing_weights[:, None, :]).sum(
                dim=2, keepdim=True
            )
            condition_parts.append(
                self.local_event_encoder(local_event, target_time_index)
            )
        condition = self.conditioning(
            torch.cat(
                condition_parts
                + [
                    self.block_encoder(block_features),
                    self.time_encoder(target_time_features),
                    self.origin_time_encoder(origin_time_features),
                    self.lead_encoder(lead_features),
                ],
                dim=1,
            )
        )

        current_state = torch.stack(
            [torch.log1p(origin_depth), origin_component_x, origin_component_y],
            dim=1,
        )
        if self.history_states == 2:
            if older_depth is None or older_component_x is None or older_component_y is None:
                raise ValueError("Two-state model requires the older hydraulic state")
            older_state = torch.stack(
                [torch.log1p(older_depth), older_component_x, older_component_y],
                dim=1,
            )
            history_delta = current_state - older_state
            if self.history_fusion == "adapter":
                state = current_state + self.history_adapter(history_delta)
            else:
                state = torch.cat([current_state, history_delta], dim=1)
        else:
            state = current_state

        x1 = self.enc1(torch.cat([static, mask.unsqueeze(1)], dim=1))
        x1 = x1 + self.state_encoder(state * mask.unsqueeze(1))
        if self.halo_encoder is not None:
            if hydraulic_context is None:
                raise ValueError("Halo model requires hydraulic context")
            halo_state = hydraulic_context.clone()
            halo_state[:, 0] = torch.log1p(torch.clamp_min(halo_state[:, 0], 0.0))
            if self.history_states == 2:
                if older_hydraulic_context is None:
                    raise ValueError("Two-state halo model requires older context")
                older_halo = older_hydraulic_context.clone()
                older_halo[:, 0] = torch.log1p(torch.clamp_min(older_halo[:, 0], 0.0))
                halo_state = torch.cat([halo_state, halo_state - older_halo], dim=1)
            halo_features = self.halo_encoder(halo_state)
            h = self.hydraulic_halo_cells
            x1 = x1 + halo_features[:, :, h:-h, h:-h]
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

        raw_depth_delta = self.depth_delta_head(x).squeeze(1) * mask
        raw_component_delta = self.component_delta_head(x) * mask.unsqueeze(1)
        activity_logits = None
        if self.activity_head is not None:
            activity_logits = self.activity_head(x).squeeze(1)
            activity_gate = torch.sigmoid(activity_logits) * mask
            depth_delta = raw_depth_delta * activity_gate
            component_delta = raw_component_delta * activity_gate.unsqueeze(1)
        else:
            depth_delta = raw_depth_delta
            component_delta = raw_component_delta
        if self.component_gate_head is not None:
            component_gate = torch.sigmoid(self.component_gate_head(x).squeeze(1)) * mask
            component_delta = component_delta * component_gate.unsqueeze(1)

        depth = torch.clamp_min(origin_depth + depth_delta, 0.0) * mask
        component_x = (origin_component_x + component_delta[:, 0]) * mask
        component_y = (origin_component_y + component_delta[:, 1]) * mask
        wet_logits = self.wet_head(x).squeeze(1)
        output = (
            depth,
            wet_logits,
            component_x,
            component_y,
            depth_delta,
            component_delta,
        )
        if activity_logits is not None:
            output += (activity_logits, raw_depth_delta, raw_component_delta)
        return output


def load_v4b_initialization(model: Stage2DirectLeadModel, checkpoint):
    """Initialize V5 from V4b while zeroing new conditioning columns."""

    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    loaded = []
    adapted = []
    skipped = []
    for name, value in source.items():
        if name not in target:
            skipped.append(name)
            continue
        if target[name].shape == value.shape:
            target[name] = value
            loaded.append(name)
            continue
        if (
            name == "conditioning.0.weight"
            and value.ndim == 2
            and target[name].ndim == 2
            and target[name].shape[0] == value.shape[0]
            and target[name].shape[1] > value.shape[1]
        ):
            expanded = torch.zeros_like(target[name])
            expanded[:, : value.shape[1]] = value
            target[name] = expanded
            adapted.append(name)
            continue
        skipped.append(name)
    model.load_state_dict(target)
    return loaded, adapted, skipped
