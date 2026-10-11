"""Reject known-contaminated legacy assets for new forecasting experiments."""

import json
import hashlib
from pathlib import Path

CORRECTED_CONVERSION = "numeric-order-signed-components-v1"


def check_target_metadata(metadata):
    if metadata.get("conversion_version") != CORRECTED_CONVERSION:
        raise ValueError("Targets lack verified numeric ordering and signed discharge provenance. "
                         "Legacy assets are contaminated; see docs/stage2/TARGET_DATA_AUDIT.md. "
                         "Rebuild/validate data before training or forecast evaluation.")
    if metadata.get("component_semantics") != "unit_discharge":
        raise ValueError("Corrected hydraulic targets must declare unit_discharge semantics")


def require_corrected_targets(manifest_dir):
    metadata = json.loads((Path(manifest_dir)/"dynamic_metadata.json").read_text())
    check_target_metadata(metadata)


def require_corrected_sampling(sampling_dir, manifest_dir):
    metadata = json.loads((Path(sampling_dir) / "sampling_metadata.json").read_text())
    if metadata.get("conversion_version") != CORRECTED_CONVERSION:
        raise ValueError("Sampling index is legacy or unverified; rebuild it from corrected targets")
    digest = hashlib.sha256((Path(manifest_dir) / "dynamic_manifest.parquet").read_bytes()).hexdigest()
    if metadata.get("source_manifest_sha256") != digest:
        raise ValueError("Sampling index was not built from this corrected manifest")
