"""Residual hydraulic state-transition model for the TRITON surrogate."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1_model import CausalEventEncoder, ConvBlock


class Stage1StateTransitionModel(nn.Module):
    """Advance one block-local hydraulic state by predicting residual changes."""

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
        history_states: int = 1,
        history_fusion: str = "concat",
        use_activity_gate: bool = False,
        activity_gate_initial_bias: float = 2.0,
    ):
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
        self.event_encoder = CausalEventEncoder(
            event_features, temporal_channels, event_embedding_dim, dropout, temporal_layers
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
        state_channels = (
            6
            if self.history_states == 2 and self.history_fusion == "concat"
            else 3
        )
        self.state_encoder = ConvBlock(state_channels, base_channels)
        self.history_adapter = (
            nn.Conv2d(3, 3, kernel_size=1)
            if self.history_states == 2 and self.history_fusion == "adapter"
            else None
        )
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4 + conditioning_dim, base_channels * 8)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.depth_delta_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.component_delta_head = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.wet_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.activity_head = (
            nn.Conv2d(base_channels, 1, kernel_size=1)
            if self.use_activity_gate
            else None
        )
        nn.init.zeros_(self.depth_delta_head.weight)
        nn.init.zeros_(self.depth_delta_head.bias)
        nn.init.zeros_(self.component_delta_head.weight)
        nn.init.zeros_(self.component_delta_head.bias)
        if self.activity_head is not None:
            nn.init.zeros_(self.activity_head.weight)
            nn.init.constant_(self.activity_head.bias, float(activity_gate_initial_bias))
        if self.history_adapter is not None:
            nn.init.zeros_(self.history_adapter.weight)
            nn.init.zeros_(self.history_adapter.bias)

    def forward(
        self,
        event,
        time_index,
        time_features,
        block_features,
        static,
        mask,
        previous_depth,
        previous_component_x,
        previous_component_y,
        shared_event_time: bool = False,
        older_depth=None,
        older_component_x=None,
        older_component_y=None,
    ):
        if shared_event_time:
            event_embedding = self.event_encoder(event[:1], time_index[:1]).expand(
                event.shape[0], -1
            )
        else:
            event_embedding = self.event_encoder(event, time_index)
        condition = self.conditioning(
            torch.cat(
                [
                    event_embedding,
                    self.block_encoder(block_features),
                    self.time_encoder(time_features),
                ],
                dim=1,
            )
        )

        current_state = torch.stack(
            [torch.log1p(previous_depth), previous_component_x, previous_component_y], dim=1
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
        depth = torch.clamp_min(previous_depth + depth_delta, 0.0) * mask
        component_x = (previous_component_x + component_delta[:, 0]) * mask
        component_y = (previous_component_y + component_delta[:, 1]) * mask
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


def load_transition_checkpoint_compatible(model, checkpoint):
    """Load compatible weights and expand a one-state encoder to delta history."""

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
        can_expand_history = (
            name.startswith("state_encoder")
            and value.ndim == 4
            and target[name].ndim == 4
            and target[name].shape[0] == value.shape[0]
            and target[name].shape[1] == 2 * value.shape[1]
            and target[name].shape[2:] == value.shape[2:]
        )
        if can_expand_history:
            expanded = torch.zeros_like(target[name])
            expanded[:, : value.shape[1]] = value
            target[name] = expanded
            adapted.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(target)
    return loaded, adapted, skipped


def load_timestamp_backbone(model, checkpoint):
    """Load shape-compatible timestamp-model weights without changing residual heads."""

    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    loaded = []
    skipped = []
    for name, value in source.items():
        if name in target and target[name].shape == value.shape:
            target[name] = value
            loaded.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(target)
    return loaded, skipped
