"""Train-only fixed-cohort learning and GPU-cost diagnostic for graph development.

This deliberately overfits training examples. It is NOT a generalization test.
"""

import argparse
import copy
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from stage2.data import DirectHorizonDataset
from stage2.direct import move_direct_batch, predict_direct
from stage2.graph_data import GraphNeighborhoodDataset, TileTopology, select_center_forecasts
from stage2.losses import delta_aware_loss_terms
from stage2.model import Stage2DirectLeadModel
from stage2.train_direct import build_bundle, seed_everything
from triton_core.metrics import MetricAccumulator
from stage2.target_integrity import require_corrected_targets


def cohort_statistics(batch, wet_threshold):
    """Summarize central labels; reject an all-dry or effectively static cohort."""
    _, central = select_center_forecasts((batch["depth"],), batch)
    mask = central["mask"] > 0.5
    target_wet = (central["depth"] >= wet_threshold) & mask
    origin_wet = (central["previous_depth"] >= wet_threshold) & mask
    active = target_wet | origin_wet
    change = (central["depth"] - central["previous_depth"]).abs()
    count = int(active.sum())
    return dict(valid_cells=int(mask.sum()), target_wet_cells=int(target_wet.sum()),
                origin_wet_cells=int(origin_wet.sum()),
                newly_wet_cells=int((target_wet & ~origin_wet).sum()),
                newly_dry_cells=int((origin_wet & ~target_wet & mask).sum()),
                active_depth_change_mean=float(change[active].mean()) if count else 0.0,
                context_nodes=int(batch["event"].shape[0]))


def select_training_cohort(dataset, sampling_dir, wet_threshold, seed):
    """Select one verified dynamic, wet center for each of four leads/events."""
    rng = np.random.default_rng(seed)
    events = dataset.events["event_id"].astype(str).tolist()
    event_order = rng.permutation(len(events)).tolist()
    records, batches = [], []
    for cohort_index, (lead, direction) in enumerate(((1, "filling"), (6, "draining"),
                                                    (24, "filling"), (48, "draining"))):
        accepted = False
        for event_position in event_order[cohort_index:] + event_order[:cohort_index]:
            event_id = events[event_position]
            shard = sampling_dir / "shards" / event_id / "sampling_candidates.parquet"
            frame = pd.read_parquet(shard, columns=["time_index", "anchor_block",
                                                   "wet_fraction", "local_transition_activity",
                                                   "local_transition_direction"])
            n_times = int(dataset.events.iloc[event_position]["n_times"])
            frame = frame.loc[(frame.wet_fraction >= 0.05)
                              & (frame.time_index >= 49)
                              & (frame.time_index - lead + 48 < n_times)
                              & (frame.anchor_block >= 0)
                              & (frame.anchor_block < len(dataset.block_rows))]
            if direction == "filling":
                frame = frame.loc[frame.local_transition_direction > 0]
            else:
                frame = frame.loc[frame.local_transition_direction < 0]
            frame = frame.sort_values("local_transition_activity", ascending=False)
            for row in frame.head(128).itertuples(index=False):
                key = (event_position, int(row.time_index), int(row.anchor_block), lead)
                batch = dataset.make_batch([key])
                statistics = cohort_statistics(batch, wet_threshold)
                if statistics["target_wet_cells"] < 16 or statistics["active_depth_change_mean"] < 0.01:
                    continue
                transition_cells = statistics["newly_wet_cells"] if direction == "filling" else statistics["newly_dry_cells"]
                if transition_cells < 16:
                    continue
                records.append(dict(event_id=event_id, event_position=event_position,
                                    requested_direction=direction,
                                    target_time_index=key[1], origin_time_index=key[1]-lead,
                                    block_position=key[2],
                                    block_label=int(dataset.block_rows.iloc[key[2]]["block_index"]),
                                    lead_steps=lead, **statistics))
                batches.append(batch)
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            raise RuntimeError(f"No verified wet/dynamic training center found for lead {lead}")
    return records, batches


def evaluate_cached(model, batches, loss_args, device):
    model.eval()
    accumulator = MetricAccumulator()
    losses = []
    terms = defaultdict(list)
    with torch.no_grad():
        for batch in batches:
            output, central = select_center_forecasts(predict_direct(model, batch), batch)
            values = delta_aware_loss_terms(output, central, loss_args, device)
            loss = values["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic loss")
            losses.append(float(loss))
            for name, value in values.items():
                terms[name].append(float(value))
            accumulator.update(output[0], torch.sigmoid(output[1]), output[2], output[3],
                               central, loss_args.wet_threshold, 1.0, loss_args.component_semantics)
    metrics = accumulator.finalize()
    if metrics["wet_cells"] <= 0:
        raise RuntimeError("Diagnostic evaluation contains no wet cells")
    return dict(loss=float(np.mean(losses)), metrics=metrics,
                loss_terms={name: float(np.mean(values)) for name, values in terms.items()})


def map_learning_gate(initial, final):
    """A loss-only gain cannot pass if the hydraulic maps deteriorate."""
    metrics = ("depth_wet_rmse", "component_rmse", "derived_velocity_rmse")
    return all(math.isfinite(final["metrics"][metric]) and
               final["metrics"][metric] <= initial["metrics"][metric] + 1e-8
               for metric in metrics) and final["metrics"]["wet_f1"] >= initial["metrics"]["wet_f1"]


def benchmark_training_batch(model, cpu_batch, loss_args, device):
    """Measure the four-center screening shape without changing model weights."""
    batch = move_direct_batch(dict(cpu_batch), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output, central = select_center_forecasts(predict_direct(model, batch), batch)
        loss = delta_aware_loss_terms(output, central, loss_args, device)["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite four-center benchmark loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite four-center benchmark gradient")
        optimizer.step()
        torch.cuda.synchronize()
        if iteration:
            times.append(time.perf_counter()-started)
    report = dict(central_tiles=int(batch["graph_center_mask"].sum()),
                  context_nodes=int(batch["event"].shape[0]),
                  graph_edges=int(batch["graph_edge_index"].shape[1]),
                  median_step_seconds=float(np.median(times)),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  scope="cached-input forward/loss/backward/AdamW; lr=0; frozen BN")
    del optimizer, output, central, loss, batch
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path,
                        default=Path("results/stage2_v5b_robustness_fold0/halo/seed_45"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-loss-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    if args.steps < 4 or args.learning_rate <= 0 or not 0 < args.max_loss_ratio < 1:
        raise ValueError("Invalid diagnostic steps/rate/loss-ratio gate")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic on an allocated GPU")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(4)
    device = torch.device("cuda")
    reference = json.loads((args.reference_run / "run_config.json").read_text())
    base = json.loads((Path(reference["initial_run_dir"]) / "run_config.json").read_text())
    require_corrected_targets(base["manifest_dir"])
    bundle_args = SimpleNamespace(data_split_seed=reference["data_split_seed"],
                                  test_events=reference["test_events"], batch_size=1,
                                  train_batches_per_epoch=4, selection_batches=1,
                                  selection_start_time_stride=12, max_horizon_steps=48,
                                  netcdf_chunk_cache_mb=32, max_open_netcdf_handles=4)
    started = time.perf_counter()
    bundle = build_bundle(base, bundle_args, Path(reference["sampling_index_dir"]))
    if bundle.split_events != reference["split_events"]:
        raise RuntimeError("Diagnostic split differs from reference")
    direct = DirectHorizonDataset(bundle.train_dataset, hydraulic_halo_cells=8)
    topology = TileTopology.from_raster(direct.block_index_grid, direct.block_rows["block_index"])
    dataset = GraphNeighborhoodDataset(direct, topology, 2)
    records, batches = select_training_cohort(dataset, Path(reference["sampling_index_dir"]),
                                             reference["wet_threshold"], args.seed)
    if any(record["event_id"] not in bundle.split_events["train"] for record in records):
        raise RuntimeError("Nontraining event in overfit cohort")
    (output_dir / "fixed_training_cohort.json").write_text(json.dumps(records, indent=2))
    (output_dir / "graph_topology.json").write_text(json.dumps(topology.metadata(), indent=2))
    benchmark_record = records[0]
    benchmark_centers = sorted(set([benchmark_record["block_position"],
                                   len(dataset.block_rows)//3, 2*len(dataset.block_rows)//3,
                                   len(dataset.block_rows)-1]))
    if len(benchmark_centers) != 4:
        raise RuntimeError("Four-center benchmark requires four distinct centers")
    benchmark_batch = dataset.make_batch([
        (benchmark_record["event_position"], benchmark_record["target_time_index"],
         center, benchmark_record["lead_steps"]) for center in benchmark_centers
    ])
    cohort_seconds = time.perf_counter() - started
    batches = [move_direct_batch(batch, device) for batch in batches]
    loss_args = SimpleNamespace(**base)
    loss_args.component_semantics = bundle.component_semantics
    model_config = dict(reference["model_config"], dropout=0.0)
    seed_everything(args.seed)
    common_weights = copy.deepcopy(Stage2DirectLeadModel(**model_config).state_dict())
    arms = {}
    for name, layers, mode in (("unet", 0, "neighbors"), ("local_control", 2, "self"),
                               ("neighbor_graph", 2, "neighbors")):
        seed_everything(args.seed)
        config = dict(model_config, graph_layers=layers, graph_mode=mode)
        model = Stage2DirectLeadModel(**config).to(device)
        missing = model.load_state_dict(common_weights, strict=False)
        if missing.unexpected_keys or any(not key.startswith("graph_") for key in missing.missing_keys):
            raise RuntimeError("Unexpected diagnostic initialization mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
        initial = evaluate_cached(model, batches, loss_args, device)
        graph_initial = {name: tensor.detach().clone() for name, tensor in model.named_parameters()
                         if name.startswith("graph_communication.")}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        training_start = time.perf_counter()
        history = []
        for step in range(args.steps):
            # Deliberately fixed BN statistics and disabled dropout make this
            # an optimization check, not a production training recipe.
            model.eval()
            batch = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            output, central = select_center_forecasts(predict_direct(model, batch), batch)
            loss = delta_aware_loss_terms(output, central, loss_args, device)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite learning-check loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite learning-check gradients")
            optimizer.step()
            history.append(float(loss.detach()))
            if (step + 1) % 32 == 0:
                print(f"{name}: step={step+1} loss={history[-1]:.6f}", flush=True)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_start
        peak_training_bytes = torch.cuda.max_memory_allocated()
        final = evaluate_cached(model, batches, loss_args, device)
        ratio = final["loss"] / max(initial["loss"], 1e-12)
        deltas = [float((tensor.detach()-graph_initial[key]).abs().max())
                  for key, tensor in model.named_parameters() if key in graph_initial]
        del optimizer, output, central, loss, graph_initial
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inference_seconds = []
        with torch.no_grad():
            for batch in batches * 3:
                torch.cuda.synchronize()
                inference_start = time.perf_counter()
                predict_direct(model, batch)
                torch.cuda.synchronize()
                inference_seconds.append(time.perf_counter()-inference_start)
        arms[name] = dict(model_config=config, trainable_parameters=sum(p.numel() for p in model.parameters()),
                          initial=initial, final=final, loss_ratio=ratio,
                          learning_gate_passed=math.isfinite(ratio) and ratio <= args.max_loss_ratio,
                          map_learning_gate_passed=map_learning_gate(initial, final),
                          graph_message_parameter_max_change=max(deltas, default=0),
                          optimizer_steps=args.steps, training_seconds=training_seconds,
                          seconds_per_step=training_seconds/args.steps,
                          inference_median_seconds=float(np.median(inference_seconds)),
                          inference_p90_seconds=float(np.percentile(inference_seconds, 90)),
                          peak_training_allocated_bytes=peak_training_bytes,
                          peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),
                          loss_history=history)
        arms[name]["four_center_training_benchmark"] = benchmark_training_batch(
            model, benchmark_batch, loss_args, device
        )
        torch.save(dict(model_config=config, model_state_dict=model.state_dict()), output_dir/f"{name}.pt")
        print(f"{name}: initial={initial['loss']:.6f} final={final['loss']:.6f} ratio={ratio:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    passed = all(arm["learning_gate_passed"] for arm in arms.values())
    passed = passed and arms["neighbor_graph"]["graph_message_parameter_max_change"] > 0
    promotion_ready = passed and all(arm["map_learning_gate_passed"] for arm in arms.values())
    report = dict(status="passed" if promotion_ready else "learning_gate_not_met",
                  loss_reduction_gate_passed=passed, promotion_ready=promotion_ready,
                  scope="train-only overfit and GPU cost; NOT forecasting generalization",
                  reference_run=str(args.reference_run.resolve()), split_events=bundle.split_events,
                  seed=args.seed, cohort_records=records, cohort_preparation_seconds=cohort_seconds,
                  four_center_benchmark_positions=benchmark_centers,
                  max_loss_ratio=args.max_loss_ratio, batch_norm="frozen initial statistics",
                  dropout=0, hardware=dict(host=platform.node(), gpu=torch.cuda.get_device_name(0),
                                           torch=torch.__version__, cuda_build=torch.version.cuda),
                  topology_sha256=topology.digest, arms=arms)
    (output_dir / "LEARNING_CHECK_RESULTS.json").write_text(json.dumps(report, indent=2))
    print(f"Learning-check status: {report['status']}; report: {output_dir}", flush=True)
    if not promotion_ready:
        raise RuntimeError("Learning check failed; do not promote to full comparison")


if __name__ == "__main__":
    main()
