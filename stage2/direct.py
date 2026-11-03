"""Shared training and evaluation operations for direct Stage-2 forecasts."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate

from stage2.losses import delta_aware_loss_terms
from stage2.depth_diagnostics import DepthDiagnostics
from stage2.graph_data import select_center_forecasts
from triton_core.metrics import MetricAccumulator
from triton_core.runtime import physical_score


DEVICE_FIELDS = (
    "graph_edge_index", "graph_group_ids", "graph_edge_attributes",
    "graph_center_mask", "graph_block_positions", "graph_context_truncated",
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
    "forcing_weights",
    "hydraulic_context",
    "older_hydraulic_context",
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
    graph_enabled = getattr(model, "graph_communication", None) is not None
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
        forcing_weights=batch.get("forcing_weights"),
        hydraulic_context=batch.get("hydraulic_context"),
        older_hydraulic_context=batch.get("older_hydraulic_context"),
        graph_edge_index=batch.get("graph_edge_index") if graph_enabled else None,
        graph_group_ids=batch.get("graph_group_ids") if graph_enabled else None,
        graph_edge_attributes=batch.get("graph_edge_attributes") if graph_enabled else None,
    )


def apply_component_policy(output, batch, policy, wet_threshold):
    """Apply a declared discharge policy while retaining learned depth/extent."""
    if policy == "learned":
        return output[2], output[3]
    if policy == "component_persistence":
        return batch["previous_component_x"], batch["previous_component_y"]
    if policy != "velocity_persistence":
        raise ValueError(f"Unsupported component policy: {policy}")
    origin_depth = batch["previous_depth"]
    origin_wet = origin_depth >= float(wet_threshold)
    denominator = origin_depth.clamp_min(float(wet_threshold))
    velocity_x = torch.where(
        origin_wet, batch["previous_component_x"] / denominator, 0.0
    )
    velocity_y = torch.where(
        origin_wet, batch["previous_component_y"] / denominator, 0.0
    )
    predicted_depth = output[0]
    return velocity_x * predicted_depth, velocity_y * predicted_depth


def extent_score(output, mode, wet_threshold):
    probability = torch.sigmoid(output[1])
    depth_wet = (output[0] >= float(wet_threshold)).to(probability.dtype)
    if mode == "head":
        return probability
    if mode == "depth":
        return depth_wet
    if mode == "or":
        return torch.maximum(probability, depth_wet)
    if mode == "and":
        return probability * depth_wet
    raise ValueError(f"Unsupported wet classification mode: {mode}")


def depth_extent_metrics_only(value):
    """Do not present compatibility discharge heads as trained outputs."""
    if not isinstance(value, dict):
        return value
    result={key:depth_extent_metrics_only(item) for key,item in value.items()
            if not any(token in key for token in ("component", "velocity", "physical_score"))}
    if "depth_wet_rmse" in result and "wet_f1" in result:
        result["depth_extent_score"]=result["depth_wet_rmse"] + .5*(1-result["wet_f1"])
    return result


def run_direct_train_epoch(model, loader, device, loss_args, optimizer):
    model.train()
    totals = defaultdict(float)
    metrics = MetricAccumulator()
    samples = 0
    horizon_counts = defaultdict(int)
    local_gradient_max = 0.0
    for batch in loader:
        batch = move_direct_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = predict_direct(model, batch)
        output, batch = select_center_forecasts(output, batch)
        values = delta_aware_loss_terms(output, batch, loss_args, device)
        values["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite V5 gradient norm")
        if getattr(model,"local_event_encoder",None) is not None:
            local_norm=sum(float(p.grad.detach().square().sum()) for p in model.local_event_encoder.parameters() if p.grad is not None) ** .5
            local_gradient_max=max(local_gradient_max,local_norm)
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
    if getattr(model,"local_event_encoder",None) is not None:
        result["local_forcing_gradient_norm_max"] = local_gradient_max
    result["horizon_sample_counts"] = {
        str(key): value for key, value in sorted(horizon_counts.items())
    }
    return depth_extent_metrics_only(result) if getattr(loss_args,"depth_extent_only",False) else result


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
    wet_probability_threshold: float = 0.5,
    wet_classification_mode: str = "head",
    component_policy: str = "learned",
    return_per_event: bool = False,
    target_batches=None,
    allow_partial_target_history=False,
    depth_extent_only=False,
    depth_diagnostics=False,
):
    """Evaluate shared origins, or explicit fixed targets with lead-specific origins."""

    horizons = sorted(set(int(value) for value in horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("All direct-evaluation horizons must be positive")
    if horizons[-1] > dataset.max_horizon_steps:
        raise ValueError("Evaluation horizon exceeds the dataset configuration")
    selected = target_batches if target_batches is not None else select_forecast_batches(
        n_times=dataset.events["n_times"].astype(int).tolist(),
        n_blocks=len(dataset.block_rows),
        batch_size=batch_size,
        max_horizon=max(horizons),
        time_stride=start_time_stride,
        batch_limit=batch_limit,
        minimum_start_time=dataset.history_states - 1,
    )
    lead_counts = {h: dict(batches=0, tile_target_pairs=0, event_ids=set()) for h in horizons}
    event_lead_counts = {}
    for event_position, time_index, blocks in selected:
        event_id = str(dataset.events.iloc[event_position]["event_id"])
        for horizon in horizons:
            if target_batches is not None and time_index - horizon < dataset.history_states - 1:
                if allow_partial_target_history:
                    continue
                raise ValueError("Target lacks history for requested forecast lead")
            counts = lead_counts[horizon]
            counts["batches"] += 1
            counts["tile_target_pairs"] += len(blocks)
            counts["event_ids"].add(event_id)
            event_lead_counts[(event_id, horizon)] = event_lead_counts.get((event_id, horizon), 0) + len(blocks)
    if any(counts["batches"] == 0 for counts in lead_counts.values()):
        raise ValueError("No eligible cases for a requested forecast lead")
    accumulators = {
        horizon: {
            "direct": MetricAccumulator(),
            "persistence": MetricAccumulator(),
        }
        for horizon in horizons
    }
    change_statistics = {h: dict(valid_cells=0, origin_wet_cells=0, target_wet_cells=0,
        newly_wet_cells=0, newly_dry_cells=0, wet_union_cells=0, depth_change_sq_sum=0.0) for h in horizons}
    per_event_accumulators = {}
    diagnostics = {}
    event_diagnostics = {}
    if depth_diagnostics:
        diagnostics = {h:{arm:DepthDiagnostics() for arm in ('direct','persistence')} for h in horizons}
    if return_per_event:
        for event_id in dataset.events["event_id"].astype(str):
            per_event_accumulators[event_id] = {
                horizon: {"direct": MetricAccumulator(), "persistence": MetricAccumulator()}
                for horizon in horizons
            }
    model.eval()
    with torch.no_grad():
        for event_position, origin_time, block_positions in selected:
            for horizon in horizons:
                target_time = int(origin_time) if target_batches is not None else int(origin_time) + int(horizon)
                if target_batches is not None and target_time - horizon < dataset.history_states - 1:
                    continue
                keys = [(event_position, target_time, block_position, horizon)
                        for block_position in block_positions]
                if hasattr(dataset, "make_batch"):
                    batch = dataset.make_batch(keys)
                else:
                    batch = default_collate([dataset[key] for key in keys])
                batch = move_direct_batch(batch, device)
                output = predict_direct(model, batch)
                output, batch = select_center_forecasts(output, batch)
                if depth_diagnostics:
                    event_id=str(dataset.events.iloc[event_position]['event_id'])
                    event_key=(event_id,horizon)
                    pair=event_diagnostics.setdefault(event_key,{arm:DepthDiagnostics() for arm in ('direct','persistence')})
                    for arm,pred in (('direct',output[0]),('persistence',batch['previous_depth'])):
                        for acc in (diagnostics[horizon][arm],pair[arm]):
                            acc.update(pred,batch['depth'],batch['previous_depth'],batch['mask'])
                component_x, component_y = apply_component_policy(
                    output, batch, component_policy, wet_threshold
                )
                predicted_extent = extent_score(
                    output, wet_classification_mode, wet_threshold
                )
                accumulators[horizon]["direct"].update(
                    output[0],
                    predicted_extent,
                    component_x,
                    component_y,
                    batch,
                    wet_threshold,
                    1.0,
                    component_semantics,
                    wet_probability_threshold,
                )
                if target_batches is not None:
                    valid = batch["mask"] > 0.5
                    ow = (batch["previous_depth"] >= wet_threshold) & valid
                    tw = (batch["depth"] >= wet_threshold) & valid
                    union = ow | tw
                    stats = change_statistics[horizon]
                    for name, cells in (("valid_cells", valid), ("origin_wet_cells", ow),
                                        ("target_wet_cells", tw), ("newly_wet_cells", tw & ~ow),
                                        ("newly_dry_cells", ow & ~tw), ("wet_union_cells", union)):
                        stats[name] += int(cells.sum().item())
                    stats["depth_change_sq_sum"] += float(((batch["depth"] - batch["previous_depth"])[union] ** 2).sum().item())
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
                    wet_probability_threshold,
                )
                if return_per_event:
                    event_id = str(dataset.events.iloc[event_position]["event_id"])
                    per_event_accumulators[event_id][horizon]["direct"].update(
                        output[0], predicted_extent, component_x, component_y, batch,
                        wet_threshold, 1.0, component_semantics,
                        wet_probability_threshold,
                    )
                    per_event_accumulators[event_id][horizon]["persistence"].update(
                        batch["previous_depth"], persistence_wet,
                        batch["previous_component_x"], batch["previous_component_y"],
                        batch, wet_threshold, 1.0, component_semantics,
                        wet_probability_threshold,
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
    if target_batches is not None:
        for horizon, stats in change_statistics.items():
            stats["depth_change_rms_wet_union_m"] = (stats["depth_change_sq_sum"] / max(stats["wet_union_cells"], 1)) ** 0.5
            result[str(horizon)]["observed_origin_to_target_change"] = stats
            counts = lead_counts[horizon]
            result[str(horizon)]["cohort_coverage"] = dict(batches=counts["batches"],
                tile_target_pairs=counts["tile_target_pairs"], event_ids=sorted(counts["event_ids"]),
                events=len(counts["event_ids"]))
    if not return_per_event:
        if depth_diagnostics:
            for h in horizons:
                result[str(h)]['depth_diagnostics']={arm:acc.finalize() for arm,acc in diagnostics[h].items()}
        return (depth_extent_metrics_only(result) if depth_extent_only else result), len(selected)
    per_event = {
        event_id: {
            str(horizon): {
                "lead_hours": horizon * dataset.step_hours,
                "tile_target_pairs": event_lead_counts.get((event_id, horizon), 0),
                "direct": _finalize(values[horizon]["direct"]) if event_lead_counts.get((event_id, horizon), 0) else None,
                "persistence": _finalize(values[horizon]["persistence"]) if event_lead_counts.get((event_id, horizon), 0) else None,
            }
            for horizon in horizons
        }
        for event_id, values in per_event_accumulators.items()
    }
    if depth_diagnostics:
        for h in horizons:
            result[str(h)]['depth_diagnostics']={arm:acc.finalize() for arm,acc in diagnostics[h].items()}
        for (event_id,h),pair in event_diagnostics.items():
            per_event[event_id][str(h)]['depth_diagnostics']={arm:acc.finalize() for arm,acc in pair.items()}
    if depth_extent_only:
        result=depth_extent_metrics_only(result)
        per_event=depth_extent_metrics_only(per_event)
    return result, len(selected), per_event


def horizon_selection_score(metrics: Dict[str, dict], velocity_weight: float = 1.0, depth_extent_only=False):
    """Average the fixed validation-horizon score used for checkpointing."""

    scores = []
    for values in metrics.values():
        direct = values["direct"]
        if depth_extent_only:
            scores.append(direct["depth_wet_rmse"]+.5*(1-direct["wet_f1"]))
            continue
        scores.append(
            direct["physical_score"]
            + float(velocity_weight) * direct.get("derived_velocity_rmse", 0.0)
        )
    return float(np.mean(scores))
