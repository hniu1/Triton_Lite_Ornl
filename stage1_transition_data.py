"""State-transition data adapters built on the existing Stage-1 data layer."""

from typing import Iterator, List

import torch
from torch.utils.data import Dataset, Sampler

from stage1_data import SampleKey


def minimum_sequence_target_time(steps: int, history_states: int = 1) -> int:
    """Return the earliest target index with a complete rollout history."""
    if steps < 1:
        raise ValueError("Transition sequence must contain at least one step")
    if history_states not in (1, 2):
        raise ValueError("Transition history must contain one or two states")
    return int(steps) + int(history_states) - 1


class Stage1TransitionDataset(Dataset):
    """Return target state at ``t`` together with hydraulic state at ``t-lag``."""

    def __init__(
        self, base_dataset: Dataset, lag: int = 1, history_states: int = 1
    ) -> None:
        if lag < 1:
            raise ValueError("Transition lag must be at least one timestep")
        if history_states not in (1, 2):
            raise ValueError("Transition history must contain one or two states")
        self.base_dataset = base_dataset
        self.lag = int(lag)
        self.history_states = int(history_states)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, key: SampleKey):
        event_position, target_time, block_position = (int(value) for value in key)
        required_history = self.lag * self.history_states
        if target_time < required_history:
            raise IndexError(
                f"Target time {target_time} has fewer than "
                f"{self.history_states} history states for lag {self.lag}"
            )
        target = self.base_dataset[(event_position, target_time, block_position)]
        previous = self.base_dataset[
            (event_position, target_time - self.lag, block_position)
        ]
        target["previous_depth"] = previous["depth"]
        target["previous_component_x"] = previous["component_x"]
        target["previous_component_y"] = previous["component_y"]
        target["previous_time_index"] = previous["time_index"]
        if self.history_states == 2:
            older = self.base_dataset[
                (event_position, target_time - 2 * self.lag, block_position)
            ]
            target["older_depth"] = older["depth"]
            target["older_component_x"] = older["component_x"]
            target["older_component_y"] = older["component_y"]
            target["older_time_index"] = older["time_index"]
        return target


class Stage1TransitionSequenceDataset(Dataset):
    """Return a sequence ending at the sampled target for rollout fine-tuning."""

    def __init__(
        self, base_dataset: Dataset, steps: int = 6, history_states: int = 1
    ) -> None:
        minimum_sequence_target_time(steps, history_states)
        self.base_dataset = base_dataset
        self.steps = int(steps)
        self.history_states = int(history_states)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, key: SampleKey):
        event_position, target_time, block_position = (int(value) for value in key)
        required_previous_steps = minimum_sequence_target_time(
            self.steps, self.history_states
        )
        if target_time < required_previous_steps:
            raise IndexError(
                f"Target time {target_time} cannot end a {self.steps}-step "
                f"sequence with {self.history_states} history states"
            )
        states = [
            self.base_dataset[(event_position, time_index, block_position)]
            for time_index in range(
                target_time - required_previous_steps, target_time + 1
            )
        ]
        target = states[-1]
        target["sequence_depth"] = torch.stack(
            [state["depth"] for state in states], dim=0
        )
        target["sequence_component_x"] = torch.stack(
            [state["component_x"] for state in states], dim=0
        )
        target["sequence_component_y"] = torch.stack(
            [state["component_y"] for state in states], dim=0
        )
        target["sequence_time_index"] = torch.stack(
            [state["time_index"] for state in states[self.history_states :]], dim=0
        )
        target["sequence_time_features"] = torch.stack(
            [state["time_features"] for state in states[self.history_states :]], dim=0
        )
        target["sequence_history_states"] = torch.tensor(self.history_states)
        return target


class TransitionBatchSampler(Sampler[List[SampleKey]]):
    """Filter an existing batch sampler to targets with an available history state."""

    def __init__(self, base_sampler: Sampler[List[SampleKey]], lag: int = 1) -> None:
        if lag < 1:
            raise ValueError("Transition lag must be at least one timestep")
        self.base_sampler = base_sampler
        self.lag = int(lag)

    def __len__(self) -> int:
        return len(self.base_sampler)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[List[SampleKey]]:
        for batch in self.base_sampler:
            if batch and int(batch[0][1]) >= self.lag:
                yield batch
