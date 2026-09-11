"""Data adapters for direct lead-time-conditioned hydraulic forecasts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


DirectForecastKey = Tuple[int, int, int, int]


def lead_time_features(lead_steps: int, max_horizon_steps: int) -> torch.Tensor:
    """Encode a forecast lead without tying the model to one specific horizon."""

    if lead_steps < 1:
        raise ValueError("Forecast lead must be at least one timestep")
    if max_horizon_steps < lead_steps:
        raise ValueError("Forecast lead exceeds the configured maximum horizon")
    fraction = float(lead_steps) / float(max_horizon_steps)
    return torch.tensor(
        [
            fraction,
            math.log1p(lead_steps) / math.log1p(max_horizon_steps),
            math.sin(math.pi * fraction),
            math.cos(math.pi * fraction),
        ],
        dtype=torch.float32,
    )


class DirectHorizonDataset(Dataset):
    """Pair an origin state with one target at an explicit future lead.

    The base dataset key is ``(event, time, block)``. This adapter accepts
    ``(event, target_time, block, lead_steps)`` and reads only the target,
    origin, and optional origin-minus-one state. Intermediate hydraulic states
    are deliberately not exposed to the model.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        max_horizon_steps: int = 48,
        step_hours: float = 0.5,
        history_states: int = 2,
        hydraulic_halo_cells: int = 0,
        forcing_weights_path: Path | None = None,
    ) -> None:
        if max_horizon_steps < 1:
            raise ValueError("Maximum forecast horizon must be positive")
        if step_hours <= 0:
            raise ValueError("Timestep hours must be positive")
        if history_states not in (1, 2):
            raise ValueError("History must contain one or two states")
        self.base_dataset = base_dataset
        self.max_horizon_steps = int(max_horizon_steps)
        self.step_hours = float(step_hours)
        self.history_states = int(history_states)
        self.hydraulic_halo_cells = int(hydraulic_halo_cells)
        if self.hydraulic_halo_cells < 0:
            raise ValueError("Hydraulic halo cannot be negative")
        self.forcing_weights = (
            np.load(Path(forcing_weights_path), mmap_mode="r")
            if forcing_weights_path is not None
            else None
        )
        if self.forcing_weights is not None and self.forcing_weights.shape[1] != 300:
            raise ValueError("Forcing weights must have shape [tiles, 300]")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, key: DirectForecastKey):
        event_position, target_time, block_position, lead_steps = (
            int(value) for value in key
        )
        if not 1 <= lead_steps <= self.max_horizon_steps:
            raise IndexError(
                f"Lead {lead_steps} is outside [1, {self.max_horizon_steps}]"
            )
        origin_time = target_time - lead_steps
        earliest_origin = self.history_states - 1
        if origin_time < earliest_origin:
            raise IndexError(
                f"Origin time {origin_time} lacks {self.history_states} history states"
            )

        target = self.base_dataset[(event_position, target_time, block_position)]
        origin = self.base_dataset[(event_position, origin_time, block_position)]
        target["previous_depth"] = origin["depth"]
        target["previous_component_x"] = origin["component_x"]
        target["previous_component_y"] = origin["component_y"]
        # The delta-aware objective uses this field to classify total change
        # over the requested lead, rather than an unobserved intermediate step.
        target["true_previous_depth"] = origin["depth"]
        target["true_previous_component_x"] = origin["component_x"]
        target["true_previous_component_y"] = origin["component_y"]
        target["origin_time_index"] = origin["time_index"]
        target["origin_time_features"] = origin["time_features"]
        target["lead_steps"] = torch.tensor(lead_steps, dtype=torch.long)
        target["lead_hours"] = torch.tensor(
            lead_steps * self.step_hours, dtype=torch.float32
        )
        target["lead_features"] = lead_time_features(
            lead_steps, self.max_horizon_steps
        )
        if self.forcing_weights is not None:
            block_index = int(
                self.base_dataset.block_rows.iloc[block_position]["block_index"]
            )
            target["forcing_weights"] = torch.from_numpy(
                np.asarray(self.forcing_weights[block_index], dtype=np.float32).copy()
            )
        if self.hydraulic_halo_cells:
            target["hydraulic_context"] = self.base_dataset.hydraulic_context(
                event_position, origin_time, block_position, self.hydraulic_halo_cells
            )
        if self.history_states == 2:
            older = self.base_dataset[
                (event_position, origin_time - 1, block_position)
            ]
            target["older_depth"] = older["depth"]
            target["older_component_x"] = older["component_x"]
            target["older_component_y"] = older["component_y"]
            target["older_time_index"] = older["time_index"]
            if self.hydraulic_halo_cells:
                target["older_hydraulic_context"] = self.base_dataset.hydraulic_context(
                    event_position,
                    origin_time - 1,
                    block_position,
                    self.hydraulic_halo_cells,
                )
        return target


class RandomLeadTimeBatchSampler(Sampler[List[DirectForecastKey]]):
    """Attach one reproducible random lead to every same-time spatial batch."""

    def __init__(
        self,
        base_sampler: Sampler,
        horizons: Sequence[int],
        seed: int,
    ) -> None:
        clean = sorted(set(int(value) for value in horizons))
        if not clean or clean[0] < 1:
            raise ValueError("At least one positive forecast horizon is required")
        self.base_sampler = base_sampler
        self.horizons = clean
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base_sampler)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[List[DirectForecastKey]]:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        for batch in self.base_sampler:
            if not batch:
                continue
            lead = int(rng.choice(self.horizons))
            yield [
                (int(event), int(target), int(block), lead)
                for event, target, block in batch
            ]
