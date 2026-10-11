"""Fail a graph smoke job if the saved graph path was not actually trained."""

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    topology = json.loads((args.run_dir / "graph_topology.json").read_text())
    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    projection = checkpoint["model_state_dict"]["graph_projection.weight"]
    assert config["graph_layers"] > 0 and config["graph_hops"] >= config["graph_layers"]
    assert config["graph_topology_sha256"] == topology["topology_sha256"]
    assert topology["directed_edges"] > 0, "No tile communication edges"
    assert torch.isfinite(projection).all() and torch.count_nonzero(projection) > 0, "Graph projection did not learn"
    results = json.loads((args.run_dir / "val_dense_24h_direct_metrics.json").read_text())
    for metrics in results["metrics"].values():
        assert math.isfinite(metrics["direct"]["physical_score"]), "Nonfinite evaluation"
    report = dict(status="passed", scope="pipeline smoke only, not forecasting skill",
                  topology_sha256=topology["topology_sha256"],
                  nodes=topology["nodes"], directed_edges=topology["directed_edges"],
                  graph_projection_abs_max=float(projection.abs().max()))
    (args.run_dir / "GRAPH_SMOKE_STATUS.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
