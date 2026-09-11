"""Shared training and evaluation operations for direct Stage-2 forecasts."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate

from stage2.losses import delta_aware_loss_terms
from triton_core.metrics import MetricAccumulator
from triton_core.runtime import physical_score


DEVICE_FIELDS = (
    "event",
    "time_index",
    "time_features",
    "origin_time_index",
    "origin_time_features",
    "lead_steps",
    "lead_hours",
    "lead_features",
    "block_features",
    "static",
    "mask",
    "depth",
    "component_x",
    "component_y",
    "previous_depth",
    "previous_component_x",
    "previous_component_y",
    "true_previous_depth",
    "true_previous_component_x",
    "true_previous_component_y",
    "older_depth",
    "older_component_x",
    "older_component_y",
    "older_time_index",
)


def select_forecast_batches(
    n_times,
    n_blocks,
    batch_size,
    max_horizon,
    time_stride,
    batch_limit,
    minimum_start_time=0,
):
    """Select deterministic max-horizon-eligible forecast-origin batches."""

    candidates = []
    for event_position, event_times in enumerate(n_times):
        latest_start = int(event_times) - int(max_horizon) - 1
        for start_time in range(
            int(minimum_start_time), latest_start + 1, int(time_stride)
        ):
            for block_start in range(0, int(n_blocks), int(batch_size)):
                candidates.append(
                    (
                        int(event_position),
                        int(start_time),
                        tuple(
                            range(
                                block_start,
                                min(block_start + int(batch_size), int(n_blocks)),
                            )
                        ),
                    )
                )
    if not candidates:
        raise ValueError("No valid batches for the requested maximum horizon")
    if batch_limit <= 0 or batch_limit >= len(candidates):
        return candidates
    selected = np.linspace(0, len(candidates) - 1, int(batch_limit), dtype=np.int64)
    return [candidates[int(index)] for index in selected]


def move_direct_batch(batch, device):
    for key in DEVICE_FIELDS:
        if key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def predict_direct(model, batch):
    return model(
        batch["event"],
        batch["time_index"],
        batch["time_features"],
        batch["origin_time_features"],
        batch["lead_features"],
        batch["block_features"],
        batch["static"],
        batch["mask"],
        batch["previous_depth"],
        batch["previous_component_x"],
        batch["previous_component_y"],
        shared_event_time=True,
        older_depth=batch.get("older_depth"),
        older_component_x=batch.get("older_component_x"),
        older_component_y=batch.get("older_component_y"),
    )


def run_direct_train_epoch(model, loader, device, loss_args, optimizer):
    model.train()
    totals = defaultdict(float)
    metrics = MetricAccumulator()
    samples = 0
    horizon_counts = defaultdict(int)
    for batch in loader:
        batch = move_direct_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = predict_direct(model, batch)
        values = delta_aware_loss_terms(output, batch, loss_args, device)
        values["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite V5 gradient norm")
        optimizer.step()

        count = int(batch["event"].shape[0])
        samples += count
        for name, value in values.items():
            totals[name] += float(value.detach()) * count
        for lead in batch["lead_steps"].detach().cpu().tolist():
            horizon_counts[int(lead)] += 1
        metrics.update(
            output[0].detach(),
            torch.sigmoid(output[1].detach()),
            output[2].detach(),
            output[3].detach(),
            batch,
            loss_args.wet_threshold,
            1.0,
            loss_args.component_semantics,
        )
    result = metrics.finalize()
    for name, total in totals.items():
        result[f"loss_{name}" if name != "loss" else "loss"] = total / max(
            samples, 1
        )
    result["physical_score"] = physical_score(result)
    result["horizon_sample_counts"] = {
        str(key): value for key, value in sorted(horizon_counts.items())
    }
    return result


def _finalize(accumulator):
    values = accumulator.finalize()
    values["physical_score"] = physical_score(values)
    return values


def evaluate_direct_horizons(
    model,
    dataset,
    device,
    horizons: Sequence[int],
    batch_size: int,
    batch_limit: int,
    start_time_stride: int,
    wet_threshold: float,
    component_semantics: str,
):
    """Evaluate every lead on one max-horizon-eligible origin cohort."""

    horizons = sorted(set(int(value) for value in horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("All direct-evaluation horizons must be positive")
    if horizons[-1] > dataset.max_horizon_steps:
        raise ValueError("Evaluation horizon exceeds the dataset configuration")
    selected = select_forecast_batches(
        n_times=dataset.events["n_times"].astype(int).tolist(),
        n_blocks=len(dataset.block_rows),
        batch_size=batch_size,
        max_horizon=max(horizons),
        time_stride=start_time_stride,
        batch_limit=batch_limit,
        minimum_start_time=dataset.history_states - 1,
    )
    accumulators = {
        horizon: {
            "direct": MetricAccumulator(),
            "persistence": MetricAccumulator(),
        }
        for horizon in horizons
    }
    model.eval()
    with torch.no_grad():
        for event_position, origin_time, block_positions in selected:
            for horizon in horizons:
                target_time = int(origin_time) + int(horizon)
                batch = default_collate(
                    [
                        dataset[
                            (
                                event_position,
                                target_time,
                                block_position,
                                horizon,
                            )
                        ]
                        for block_position in block_positions
                    ]
                )
                batch = move_direct_batch(batch, device)
                output = predict_direct(model, batch)
                accumulators[horizon]["direct"].update(
                    output[0],
                    torch.sigmoid(output[1]),
                    output[2],
                    output[3],
                    batch,
                    wet_threshold,
                    1.0,
                    component_semantics,
                )
                persistence_wet = (
                    batch["previous_depth"] >= wet_threshold
                ).float()
                accumulators[horizon]["persistence"].update(
                    batch["previous_depth"],
                    persistence_wet,
                    batch["previous_component_x"],
                    batch["previous_component_y"],
                    batch,
                    wet_threshold,
                    1.0,
                    component_semantics,
                )

    result: Dict[str, dict] = {}
    for horizon in horizons:
        direct = _finalize(accumulators[horizon]["direct"])
        persistence = _finalize(accumulators[horizon]["persistence"])
        comparison = {
            "depth_wet_rmse_change_percent": 100.0
            * (direct["depth_wet_rmse"] - persistence["depth_wet_rmse"])
            / max(persistence["depth_wet_rmse"], 1e-12),
            "component_rmse_change_percent": 100.0
            * (direct["component_rmse"] - persistence["component_rmse"])
            / max(persistence["component_rmse"], 1e-12),
            "wet_f1_absolute_change": direct["wet_f1"] - persistence["wet_f1"],
            "physical_score_change_percent": 100.0
            * (direct["physical_score"] - persistence["physical_score"])
            / max(persistence["physical_score"], 1e-12),
        }
        if "derived_velocity_rmse" in direct:
            comparison["derived_velocity_rmse_change_percent"] = 100.0 * (
                direct["derived_velocity_rmse"]
                - persistence["derived_velocity_rmse"]
            ) / max(persistence["derived_velocity_rmse"], 1e-12)
        result[str(horizon)] = {
            "lead_hours": horizon * dataset.step_hours,
            "direct": direct,
            "persistence": persistence,
            "direct_vs_persistence": comparison,
        }
    return result, len(selected)


def horizon_selection_score(metrics: Dict[str, dict], velocity_weight: float = 1.0):
    """Average the fixed validation-horizon score used for checkpointing."""

    scores = []
    for values in metrics.values():
        direct = values["direct"]
        scores.append(
            direct["physical_score"]
            + float(velocity_weight) * direct.get("derived_velocity_rmse", 0.0)
        )
    return float(np.mean(scores))
