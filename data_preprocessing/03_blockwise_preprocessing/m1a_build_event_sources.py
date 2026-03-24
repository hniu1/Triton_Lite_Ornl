import argparse
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd


LOGGER = logging.getLogger("m1a_build_event_sources")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_watershed_id(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def sanitize_event_id(raw: str) -> str:
    value = (raw or "").strip()
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    return value.strip("_")


def choose_zip_member(
    zip_file: zipfile.ZipFile,
    event_id: str,
    zip_member_template: Optional[str],
    zip_member_path: Optional[str],
) -> str:
    members = [name for name in zip_file.namelist() if not name.endswith("/")]
    if not members:
        raise ValueError("Zip archive contains no files")

    if zip_member_path:
        if zip_member_path not in members:
            raise ValueError(f"zip_member_path '{zip_member_path}' not found in archive")
        return zip_member_path

    if zip_member_template:
        candidate = zip_member_template.format(event_id=event_id)
        if candidate not in members:
            raise ValueError(f"zip_member_template resolved to '{candidate}', not found in archive")
        return candidate

    preferred = [
        name
        for name in members
        if re.search(r"input/hyg/.*\.(txt|hyg|csv)$", name, flags=re.IGNORECASE)
    ]
    if preferred:
        return sorted(preferred)[0]

    possible = [name for name in members if re.search(r"\.(txt|hyg|csv)$", name, flags=re.IGNORECASE)]
    if possible:
        return sorted(possible)[0]

    return sorted(members)[0]


def read_event_text(
    source_path: Path,
    event_id: str,
    zip_member_template: Optional[str],
    zip_member_path: Optional[str],
) -> str:
    if source_path.suffix.lower() != ".zip":
        return source_path.read_text(encoding="utf-8", errors="ignore")

    with zipfile.ZipFile(source_path, "r") as zf:
        member = choose_zip_member(
            zip_file=zf,
            event_id=event_id,
            zip_member_template=zip_member_template,
            zip_member_path=zip_member_path,
        )
        LOGGER.info("Reading zip member '%s' from %s", member, source_path)
        return zf.read(member).decode("utf-8", errors="ignore")


def parse_hyg_text_to_dataframe(raw_text: str, source_label: str) -> pd.DataFrame:
    content = "\n".join(
        line for line in raw_text.splitlines() if line.strip() and not line.strip().startswith("%")
    )
    if not content.strip():
        raise ValueError(f"No parseable content found in {source_label}")

    first_line = content.splitlines()[0]
    has_header = bool(re.search(r"[A-Za-z]", first_line))
    delimiter = "," if "," in first_line else r"\s+"

    frame = pd.read_csv(
        io.StringIO(content),
        sep=delimiter,
        engine="python",
        header=0 if has_header else None,
    )

    if not has_header:
        if frame.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns in {source_label}, got {frame.shape[1]}")
        frame.columns = ["Time (hr)"] + [f"Loc{i}" for i in range(1, frame.shape[1])]

    return frame


def build_records_from_manifest(input_dir: Path, manifest_csv: Path) -> List[dict]:
    frame = pd.read_csv(manifest_csv)
    required = {"event_id", "watershed_id", "hyg_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"event_manifest_csv missing columns: {sorted(missing)}")

    records: List[dict] = []
    for _, row in frame.iterrows():
        source_path = Path(row["hyg_path"])
        if not source_path.is_absolute():
            source_path = (input_dir / source_path).resolve()

        records.append(
            {
                "event_id": sanitize_event_id(str(row["event_id"])),
                "watershed_id": normalize_watershed_id(str(row["watershed_id"])),
                "source_path": source_path,
                "zip_member_path": (
                    str(row["zip_member_path"]) if "zip_member_path" in frame.columns and pd.notna(row["zip_member_path"]) else ""
                ),
            }
        )
    return records


def build_records_from_glob(input_dir: Path, file_pattern: str, default_watershed_id: str) -> List[dict]:
    files = sorted(input_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{file_pattern}' under {input_dir}")

    watershed_id = normalize_watershed_id(default_watershed_id)
    if not watershed_id:
        raise ValueError("default_watershed_id is required when event_manifest_csv is not provided")

    records = []
    for file_path in files:
        records.append(
            {
                "event_id": sanitize_event_id(file_path.stem),
                "watershed_id": watershed_id,
                "source_path": file_path.resolve(),
                "zip_member_path": "",
            }
        )
    return records


def materialize_event_csv(
    record: dict,
    out_dir: Path,
    zip_member_template: Optional[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    event_id = record["event_id"]
    source_path: Path = record["source_path"]
    zip_member_path = record.get("zip_member_path") or None

    raw_text = read_event_text(
        source_path=source_path,
        event_id=event_id,
        zip_member_template=zip_member_template,
        zip_member_path=zip_member_path,
    )
    frame = parse_hyg_text_to_dataframe(raw_text, str(source_path))

    out_path = out_dir / f"{event_id}.csv"
    frame.to_csv(out_path, index=False)
    LOGGER.info("Materialized event_id=%s to %s shape=%s", event_id, out_path, frame.shape)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 1A: Build event source manifest from raw hyg inputs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", type=str, default="*.zip")
    parser.add_argument("--event-manifest-csv", type=Path, default=None)
    parser.add_argument("--default-watershed-id", type=str, default=None)
    parser.add_argument("--zip-member-template", type=str, default=None)
    parser.add_argument("--materialize-csv-dir", type=Path, default=None)
    parser.add_argument("--output-events-source-csv", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    if args.event_manifest_csv is not None:
        records = build_records_from_manifest(args.input_dir, args.event_manifest_csv)
    else:
        records = build_records_from_glob(args.input_dir, args.file_pattern, args.default_watershed_id)

    rows = []
    for record in records:
        source_path: Path = record["source_path"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")

        output_path = source_path
        if args.materialize_csv_dir is not None:
            output_path = materialize_event_csv(record, args.materialize_csv_dir, args.zip_member_template)

        rows.append(
            {
                "event_id": record["event_id"],
                "watershed_id": record["watershed_id"],
                "hyg_path": str(output_path),
                "zip_member_path": "",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["watershed_id", "event_id"]).reset_index(drop=True)
    args.output_events_source_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_events_source_csv, index=False)
    LOGGER.info("Wrote event source table (%d rows): %s", len(out_df), args.output_events_source_csv)


if __name__ == "__main__":
    main()
