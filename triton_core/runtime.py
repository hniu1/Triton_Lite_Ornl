"""Shared device and scoring helpers."""

import torch


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def physical_score(metrics):
    """Established depth, unit-discharge, and inundation composite."""

    return (
        metrics["depth_wet_rmse"]
        + 2.0 * metrics["component_rmse"]
        + 0.5 * (1.0 - metrics["wet_f1"])
    )

