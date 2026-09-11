#!/usr/bin/env python3
"""Build sparse distance weights from ordered forcing locations to tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks-parquet", type=Path, required=True)
    parser.add_argument("--locations-shapefile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nearest-locations", type=int, default=8)
    parser.add_argument("--distance-floor-m", type=float, default=100.0)
    return parser.parse_args()


def build_weights(blocks, locations, nearest_locations, distance_floor_m):
    if nearest_locations < 1 or nearest_locations > len(locations):
        raise ValueError("nearest-locations is outside the available location count")
    xy_blocks = blocks[["centroid_x", "centroid_y"]].to_numpy(np.float64)
    xy_locations = locations[["X", "Y"]].to_numpy(np.float64)
    squared = ((xy_blocks[:, None, :] - xy_locations[None, :, :]) ** 2).sum(2)
    nearest = np.argpartition(squared, nearest_locations - 1, axis=1)[
        :, :nearest_locations
    ]
    weights = np.zeros_like(squared, dtype=np.float32)
    rows = np.arange(len(blocks))[:, None]
    distances = np.sqrt(squared[rows, nearest])
    local = 1.0 / np.maximum(distances, float(distance_floor_m))
    local /= local.sum(axis=1, keepdims=True)
    weights[rows, nearest] = local.astype(np.float32)
    return weights


def main():
    args = parse_args()
    blocks = pd.read_parquet(args.blocks_parquet).sort_values(
        ["watershed_id", "block_id"]
    ).reset_index(drop=True)
    locations = gpd.read_file(args.locations_shapefile).sort_values("Loc").reset_index(drop=True)
    expected = np.arange(1, len(locations) + 1)
    if not np.array_equal(locations["Loc"].to_numpy(dtype=int), expected):
        raise ValueError("Forcing locations must be consecutively ordered Loc1..LocN")
    weights = build_weights(
        blocks, locations, args.nearest_locations, args.distance_floor_m
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "tile_forcing_weights.npy", weights)
    metadata = {
        "n_tiles": int(weights.shape[0]),
        "n_forcing_locations": int(weights.shape[1]),
        "nearest_locations": args.nearest_locations,
        "distance_floor_m": args.distance_floor_m,
        "location_order": "columns follow ascending Loc identifiers",
        "weighting": "inverse distance normalized over K nearest locations",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {weights.shape} weights to {output}")


if __name__ == "__main__":
    main()
