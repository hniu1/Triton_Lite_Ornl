"""Transition-regime-aware samplers for hydraulic rollout fine-tuning."""

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

from stage1_data import BalancedLabelBatchSampler, LabelAwareBatchSampler


class TransitionAwareBatchSampler(BalancedLabelBatchSampler):
    """Balance event/time groups by true change regime and patch labels."""

    def __init__(
        self,
        candidates: pd.DataFrame,
        event_ids: Sequence[str],
        n_times: Sequence[int],
        n_blocks: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
        category_fractions: Dict[str, float],
        phase_fractions: Dict[str, float],
        transition_regime_fractions: Dict[str, float],
        target_wet_cell_fraction: float,
        strict_category_quotas: bool = False,
    ) -> None:
        if "transition_regime" not in candidates:
            raise ValueError("Sampling index is missing column: transition_regime")
        event_positions = {
            str(event_id): position for position, event_id in enumerate(event_ids)
        }
        regime_frame = candidates.loc[
            candidates["event_id"].astype(str).isin(event_positions),
            ["event_id", "time_index", "transition_regime"],
        ].copy()
        regime_frame["event_id"] = regime_frame["event_id"].astype(str)
        inconsistent = regime_frame.groupby(
            ["event_id", "time_index"], observed=True
        )["transition_regime"].nunique()
        if (inconsistent > 1).any():
            raise ValueError("Transition regime must be unique within each event/time group")
        regime_frame = regime_frame.drop_duplicates(["event_id", "time_index"])
        regime_lookup = {
            (event_positions[row.event_id], int(row.time_index)): str(row.transition_regime)
            for row in regime_frame.itertuples(index=False)
        }

        super().__init__(
            candidates=candidates,
            event_ids=event_ids,
            n_times=n_times,
            n_blocks=n_blocks,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            seed=seed,
            category_fractions=category_fractions,
            phase_fractions=phase_fractions,
            target_wet_cell_fraction=target_wet_cell_fraction,
            strict_category_quotas=strict_category_quotas,
        )
        self.transition_regime_fractions = LabelAwareBatchSampler._validate_fractions(
            transition_regime_fractions, "transition regime"
        )
        self.group_regimes = {
            key: regime_lookup[key] for key in self.groups if key in regime_lookup
        }
        if len(self.group_regimes) != len(self.groups):
            raise ValueError("Some feasible event/time groups lack a transition regime")
        self.regime_pools = {}
        for key, regime in self.group_regimes.items():
            event_position, _ = key
            phase = str(self.groups[key]["phase"])
            self.regime_pools.setdefault(regime, {}).setdefault(phase, {}).setdefault(
                int(event_position), []
            ).append(key)
        self.available_transition_regimes = sorted(self.regime_pools)

    def _select_group(self, rng):
        regime = LabelAwareBatchSampler._choice(
            rng,
            self.available_transition_regimes,
            self.transition_regime_fractions,
        )
        phase_pools = self.regime_pools[regime]
        phase = LabelAwareBatchSampler._choice(
            rng, sorted(phase_pools), self.phase_fractions
        )
        event_pools = phase_pools[phase]
        event_position = int(rng.choice(sorted(event_pools)))
        keys = event_pools[event_position]
        return keys[int(rng.integers(0, len(keys)))]


class LocalTransitionAwareBatchSampler(BalancedLabelBatchSampler):
    """Balance batches around exact same-patch transition candidates.

    Each selected batch contains at least one verified candidate from the
    requested transition regime while preserving category quotas and the wet
    cell target. This avoids assigning one regime to a spatially heterogeneous
    event/time group.
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        event_ids: Sequence[str],
        n_times: Sequence[int],
        n_blocks: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
        category_fractions: Dict[str, float],
        phase_fractions: Dict[str, float],
        transition_regime_fractions: Dict[str, float],
        target_wet_cell_fraction: float,
        strict_category_quotas: bool = True,
    ) -> None:
        required = {"local_transition_regime", "local_transition_activity"}
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(
                f"Paired transition index is missing columns: {sorted(missing)}"
            )
        if not strict_category_quotas:
            raise ValueError("Local transition balancing requires strict category quotas")
        super().__init__(
            candidates=candidates,
            event_ids=event_ids,
            n_times=n_times,
            n_blocks=n_blocks,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            seed=seed,
            category_fractions=category_fractions,
            phase_fractions=phase_fractions,
            target_wet_cell_fraction=target_wet_cell_fraction,
            strict_category_quotas=True,
        )
        self.transition_regime_fractions = LabelAwareBatchSampler._validate_fractions(
            transition_regime_fractions, "transition regime"
        )
        event_positions = {
            str(event_id): position for position, event_id in enumerate(event_ids)
        }
        frame = candidates.loc[
            candidates["event_id"].astype(str).isin(event_positions)
        ].copy()
        frame["event_id"] = frame["event_id"].astype(str)
        frame["event_position"] = frame["event_id"].map(event_positions).astype(np.int32)
        frame = frame.drop_duplicates(
            ["event_position", "time_index", "anchor_block"]
        )
        indexed = frame.set_index(
            ["event_position", "time_index", "anchor_block"], drop=False
        )

        self.local_regimes: Dict[Tuple[int, int], np.ndarray] = {}
        self.local_activities: Dict[Tuple[int, int], np.ndarray] = {}
        self.eligible_anchors: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {}
        self.regime_pools: Dict[str, Dict[str, Dict[int, List[Tuple[int, int]]]]] = {}
        for key, group in self.groups.items():
            event_position, time_index = key
            blocks = group["blocks"]
            rows = indexed.loc[
                [(event_position, time_index, int(block)) for block in blocks]
            ]
            regimes = rows["local_transition_regime"].astype(str).to_numpy()
            activities = rows["local_transition_activity"].to_numpy(dtype=np.float32)
            self.local_regimes[key] = regimes
            self.local_activities[key] = activities
            for regime in np.unique(regimes):
                eligible = self._eligible_anchor_indices(key, str(regime))
                if len(eligible) == 0:
                    continue
                self.eligible_anchors[(str(regime), key)] = eligible
                phase = str(group["phase"])
                self.regime_pools.setdefault(str(regime), {}).setdefault(
                    phase, {}
                ).setdefault(int(event_position), []).append(key)
        if not self.regime_pools:
            raise ValueError("No feasible groups contain a verified local transition")
        requested = set(self.transition_regime_fractions)
        unavailable = requested - set(self.regime_pools)
        if unavailable:
            raise ValueError(
                f"Requested local transition regimes are unavailable: {sorted(unavailable)}"
            )
        self.available_transition_regimes = sorted(self.regime_pools)

    def _maximum_wet_fraction_with_anchor(self, key, anchor_index: int) -> float:
        group = self.groups[key]
        categories = group["categories"]
        wet_fractions = group["wet_fractions"]
        anchor_category = str(categories[anchor_index])
        if self.category_counts.get(anchor_category, 0) <= 0:
            return -np.inf
        total = float(wet_fractions[anchor_index])
        for category, requested in self.category_counts.items():
            remaining = requested - int(category == anchor_category)
            if remaining <= 0:
                continue
            available = np.flatnonzero(categories == category)
            available = available[available != anchor_index]
            if len(available) < remaining:
                return -np.inf
            best = available[np.argsort(-wet_fractions[available])[:remaining]]
            total += float(wet_fractions[best].sum())
        return total / self.batch_size

    def _eligible_anchor_indices(self, key, regime: str) -> np.ndarray:
        matching = np.flatnonzero(self.local_regimes[key] == regime)
        if len(matching) == 0:
            return np.empty(0, dtype=np.int64)

        group = self.groups[key]
        categories = group["categories"]
        wet_fractions = group["wet_fractions"]
        top_indices = {}
        top_sums = {}
        total_top_sum = 0.0
        for category, requested in self.category_counts.items():
            if requested <= 0:
                continue
            available = np.flatnonzero(categories == category)
            ordered = available[np.argsort(-wet_fractions[available])]
            selected = ordered[:requested]
            top_indices[category] = selected
            top_sums[category] = float(wet_fractions[selected].sum())
            total_top_sum += top_sums[category]

        maximum_sums = np.full(len(matching), -np.inf, dtype=np.float64)
        matching_categories = categories[matching]
        for category, requested in self.category_counts.items():
            positions = np.flatnonzero(matching_categories == category)
            if requested <= 0 or len(positions) == 0:
                continue
            anchors = matching[positions]
            other_categories = total_top_sum - top_sums[category]
            if requested == 1:
                category_sums = wet_fractions[anchors].astype(np.float64)
            else:
                selected = top_indices[category]
                best_without_anchor = selected[: requested - 1]
                best_without_sum = float(
                    wet_fractions[best_without_anchor].sum()
                )
                selected_sum = top_sums[category]
                anchor_already_selected = np.isin(anchors, best_without_anchor)
                category_sums = np.where(
                    anchor_already_selected,
                    selected_sum,
                    wet_fractions[anchors].astype(np.float64) + best_without_sum,
                )
            maximum_sums[positions] = other_categories + category_sums

        feasible = maximum_sums / self.batch_size + 1e-8 >= self.target_wet_cell_fraction
        return matching[feasible].astype(np.int64, copy=False)

    def _select_regime_and_group(self, rng):
        regime = LabelAwareBatchSampler._choice(
            rng,
            self.available_transition_regimes,
            self.transition_regime_fractions,
        )
        phase_pools = self.regime_pools[regime]
        phase = LabelAwareBatchSampler._choice(
            rng, sorted(phase_pools), self.phase_fractions
        )
        event_pools = phase_pools[phase]
        event_position = int(rng.choice(sorted(event_pools)))
        keys = event_pools[event_position]
        return regime, keys[int(rng.integers(0, len(keys)))]

    def _select_blocks_with_verified_anchor(self, rng, regime: str, key) -> np.ndarray:
        group = self.groups[key]
        blocks = group["blocks"]
        categories = group["categories"]
        wet_fractions = group["wet_fractions"]
        eligible = self.eligible_anchors[(regime, key)]
        anchor_weights = self.local_activities[key][eligible].astype(np.float64) + 1e-6
        anchor_weights /= anchor_weights.sum()
        anchor = int(rng.choice(eligible, p=anchor_weights))
        anchor_category = str(categories[anchor])
        selected = [anchor]
        selected_set = {anchor}
        for category, requested in self.category_counts.items():
            remaining = requested - int(category == anchor_category)
            if remaining <= 0:
                continue
            available = np.asarray(
                [
                    index
                    for index in np.flatnonzero(categories == category)
                    if int(index) not in selected_set
                ],
                dtype=np.int64,
            )
            chosen = rng.choice(available, size=remaining, replace=False).tolist()
            selected.extend(int(value) for value in chosen)
            selected_set.update(int(value) for value in chosen)

        current = float(wet_fractions[selected].mean())
        if current + 1e-8 < self.target_wet_cell_fraction:
            for category in self.category_counts:
                low = sorted(
                    [
                        index
                        for index in selected
                        if index != anchor and categories[index] == category
                    ],
                    key=lambda index: wet_fractions[index],
                )
                high = sorted(
                    [
                        index
                        for index in range(len(blocks))
                        if index not in selected_set and categories[index] == category
                    ],
                    key=lambda index: wet_fractions[index],
                    reverse=True,
                )
                for low_index, high_index in zip(low, high):
                    if wet_fractions[high_index] <= wet_fractions[low_index]:
                        break
                    position = selected.index(low_index)
                    selected[position] = high_index
                    selected_set.remove(low_index)
                    selected_set.add(high_index)
                    current += float(
                        (wet_fractions[high_index] - wet_fractions[low_index])
                        / self.batch_size
                    )
                    if current + 1e-8 >= self.target_wet_cell_fraction:
                        break
                if current + 1e-8 >= self.target_wet_cell_fraction:
                    break
        if current + 1e-8 < self.target_wet_cell_fraction:
            raise RuntimeError("Verified local-transition batch missed its wet-cell target")
        if anchor not in selected:
            raise RuntimeError("Verified local-transition anchor was lost during selection")
        return np.sort(blocks[np.asarray(selected, dtype=np.int64)])

    def __iter__(self) -> Iterator[List[Tuple[int, int, int]]]:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        for _ in range(self.batches_per_epoch):
            regime, key = self._select_regime_and_group(rng)
            event_position, time_index = key
            blocks = self._select_blocks_with_verified_anchor(rng, regime, key)
            yield [
                (int(event_position), int(time_index), int(block_position))
                for block_position in blocks
            ]
