"""Hydraulic forecast metrics shared by active experiments."""

from typing import Dict

import numpy as np
import torch


class MetricAccumulator:
    """Accumulate depth, extent, unit-discharge, and derived-velocity errors."""

    def __init__(self):
        self.values = {
            "depth_all_count": 0.0,
            "depth_all_abs": 0.0,
            "depth_all_sq": 0.0,
            "depth_wet_count": 0.0,
            "depth_wet_abs": 0.0,
            "depth_wet_sq": 0.0,
            "component_count": 0.0,
            "component_abs": 0.0,
            "component_sq": 0.0,
            "velocity_count": 0.0,
            "velocity_abs": 0.0,
            "velocity_sq": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "wet_cells": 0.0,
            "valid_cells": 0.0,
            "patches": 0.0,
            "dry_patches": 0.0,
            "boundary_patches": 0.0,
            "partial_wet_patches": 0.0,
            "mostly_wet_patches": 0.0,
            "deep_patches": 0.0,
        }

    def update(
        self,
        depth,
        wet_prob,
        component_x,
        component_y,
        batch,
        wet_threshold: float,
        deep_threshold: float,
        component_semantics: str = "unknown",
    ):
        mask = batch["mask"] > 0.5
        true_wet = (batch["depth"] >= wet_threshold) & mask
        predicted_wet = (wet_prob >= 0.5) & mask
        depth_error = depth - batch["depth"]
        all_error = depth_error[mask]
        wet_error = depth_error[true_wet]
        component_error = torch.cat(
            [
                (component_x - batch["component_x"])[true_wet],
                (component_y - batch["component_y"])[true_wet],
            ]
        )
        velocity_error = None
        if component_semantics == "unit_discharge" and true_wet.any():
            true_depth = batch["depth"][true_wet].clamp_min(wet_threshold)
            predicted_depth = depth[true_wet].clamp_min(wet_threshold)
            velocity_error = torch.cat(
                [
                    component_x[true_wet] / predicted_depth
                    - batch["component_x"][true_wet] / true_depth,
                    component_y[true_wet] / predicted_depth
                    - batch["component_y"][true_wet] / true_depth,
                ]
            )

        values = self.values
        values["depth_all_count"] += all_error.numel()
        values["depth_all_abs"] += all_error.abs().sum().item()
        values["depth_all_sq"] += all_error.square().sum().item()
        values["depth_wet_count"] += wet_error.numel()
        values["depth_wet_abs"] += wet_error.abs().sum().item()
        values["depth_wet_sq"] += wet_error.square().sum().item()
        values["component_count"] += component_error.numel()
        values["component_abs"] += component_error.abs().sum().item()
        values["component_sq"] += component_error.square().sum().item()
        if velocity_error is not None:
            values["velocity_count"] += velocity_error.numel()
            values["velocity_abs"] += velocity_error.abs().sum().item()
            values["velocity_sq"] += velocity_error.square().sum().item()
        values["tp"] += (predicted_wet & true_wet).sum().item()
        values["fp"] += (predicted_wet & ~true_wet & mask).sum().item()
        values["fn"] += (~predicted_wet & true_wet).sum().item()
        values["wet_cells"] += true_wet.sum().item()
        values["valid_cells"] += mask.sum().item()
        valid_per_patch = mask.sum(dim=(1, 2)).clamp_min(1)
        wet_per_patch = true_wet.sum(dim=(1, 2))
        wet_fraction = wet_per_patch / valid_per_patch
        patch_max_depth = batch["depth"].masked_fill(~mask, 0.0).amax(dim=(1, 2))
        values["patches"] += depth.shape[0]
        values["dry_patches"] += (wet_per_patch == 0).sum().item()
        values["boundary_patches"] += (
            (wet_fraction > 0) & (wet_fraction < 0.10)
        ).sum().item()
        values["partial_wet_patches"] += (
            (wet_fraction >= 0.10) & (wet_fraction < 0.50)
        ).sum().item()
        values["mostly_wet_patches"] += (wet_fraction >= 0.50).sum().item()
        values["deep_patches"] += (patch_max_depth >= deep_threshold).sum().item()

    def finalize(self) -> Dict[str, float]:
        values = self.values
        result = {}
        for prefix in ("depth_all", "depth_wet", "component"):
            count = max(values[f"{prefix}_count"], 1.0)
            result[f"{prefix}_mae"] = values[f"{prefix}_abs"] / count
            result[f"{prefix}_rmse"] = float(
                np.sqrt(values[f"{prefix}_sq"] / count)
            )
        if values["velocity_count"] > 0:
            result["derived_velocity_mae"] = (
                values["velocity_abs"] / values["velocity_count"]
            )
            result["derived_velocity_rmse"] = float(
                np.sqrt(values["velocity_sq"] / values["velocity_count"])
            )
        precision = values["tp"] / max(values["tp"] + values["fp"], 1.0)
        recall = values["tp"] / max(values["tp"] + values["fn"], 1.0)
        result["wet_precision"] = precision
        result["wet_recall"] = recall
        result["wet_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
        result["wet_csi"] = values["tp"] / max(
            values["tp"] + values["fp"] + values["fn"], 1.0
        )
        result["wet_cells"] = values["wet_cells"]
        result["wet_cell_fraction"] = values["wet_cells"] / max(
            values["valid_cells"], 1.0
        )
        for key in (
            "dry_patches",
            "boundary_patches",
            "partial_wet_patches",
            "mostly_wet_patches",
            "deep_patches",
        ):
            result[key.replace("_patches", "_patch_fraction")] = values[key] / max(
                values["patches"], 1.0
            )
        return result

