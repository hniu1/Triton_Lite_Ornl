import argparse
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("m1b_event_to_tensor")


@dataclass
class EventRecord:
    event_id: str
    watershed_id: str
    source_path: Path
    zip_member_path: Optional[str] = None


@dataclass
class ParsedEvent:
    event_id: str
    watershed_id: str
    time_values: np.ndarray
    sensor_ids: List[int]
    values: np.ndarray


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_watershed_id(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_loc_id(column_name: str) -> Optional[int]:
    match = re.search(r"loc\s*([0-9]+)", str(column_name), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def detect_time_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if str(column).strip().lower().startswith("time"):
            return column
    return df.columns[0]


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


def read_text_event_file(
    file_path: Path,
    event_id: str,
    zip_member_template: Optional[str],
    zip_member_path: Optional[str],
) -> str:
    if file_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(file_path, "r") as zf:
            member = choose_zip_member(zf, event_id, zip_member_template, zip_member_path)
            LOGGER.info("Reading zip member '%s' from %s", member, file_path)
            return zf.read(member).decode("utf-8", errors="ignore")
    return file_path.read_text(encoding="utf-8", errors="ignore")


def parse_hyg_to_dataframe(
    file_path: Path,
    event_id: str,
    zip_member_template: Optional[str],
    zip_member_path: Optional[str],
) -> pd.DataFrame:
    raw_text = read_text_event_file(file_path, event_id, zip_member_template, zip_member_path)
    content = "\n".join(
        line for line in raw_text.splitlines() if line.strip() and not line.strip().startswith("%")
    )

    if not content.strip():
        raise ValueError(f"No parseable content found in {file_path}")

    first_line = content.splitlines()[0]
    has_header = bool(re.search(r"[A-Za-z]", first_line))
    delimiter = "," if "," in first_line else r"\s+"

    df = pd.read_csv(
        io.StringIO(content),
        sep=delimiter,
        engine="python",
        header=0 if has_header else None,
    )

    if not has_header:
        if df.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns (time + sensors), got {df.shape[1]} in {file_path}")
        df.columns = ["Time (hr)"] + [f"Loc{i}" for i in range(1, df.shape[1])]

    return df


def load_event_records(events_source_csv: Path, base_dir: Path) -> List[EventRecord]:
    frame = pd.read_csv(events_source_csv)
    required = {"event_id", "watershed_id", "hyg_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"events_source_csv missing columns: {sorted(missing)}")

    records: List[EventRecord] = []
    for _, row in frame.iterrows():
        source_path = Path(row["hyg_path"])
        if not source_path.is_absolute():
            source_path = (base_dir / source_path).resolve()

        records.append(
            EventRecord(
                event_id=str(row["event_id"]),
                watershed_id=normalize_watershed_id(str(row["watershed_id"])),
                source_path=source_path,
                zip_member_path=(
                    str(row["zip_member_path"]) if "zip_member_path" in frame.columns and pd.notna(row["zip_member_path"]) and str(row["zip_member_path"]).strip() else None
                ),
            )
        )
    return records


def load_sensor_map(
    sensor_map_csv: Optional[Path],
    watershed_column: str,
    default_watershed_id: Optional[str],
) -> Optional[pd.DataFrame]:
    if sensor_map_csv is None:
        return None

    df = pd.read_csv(sensor_map_csv)
    required = {"Loc"}
    if default_watershed_id is None:
        required.add(watershed_column)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sensor_map_csv is missing required columns: {sorted(missing)}")

    if default_watershed_id is not None:
        watershed_values = pd.Series(default_watershed_id, index=df.index, dtype="object")
    else:
        watershed_values = df[watershed_column].astype(str)

    normalized = pd.DataFrame(
        {
            "Loc": pd.to_numeric(df["Loc"], errors="coerce").astype("Int64"),
            "watershed_id": watershed_values.map(normalize_watershed_id),
        }
    ).dropna(subset=["Loc"])
    normalized["Loc"] = normalized["Loc"].astype(int)
    return normalized


def select_sensor_ids_for_watershed(
    all_sensor_ids: Sequence[int],
    watershed_id: str,
    sensor_map_df: Optional[pd.DataFrame],
) -> List[int]:
    if sensor_map_df is None:
        return list(all_sensor_ids)

    allowed = set(
        sensor_map_df.loc[sensor_map_df["watershed_id"] == normalize_watershed_id(watershed_id), "Loc"].tolist()
    )
    selected = [sensor for sensor in all_sensor_ids if sensor in allowed]
    if not selected:
        available_ids = sorted(sensor_map_df["watershed_id"].dropna().astype(str).unique().tolist())
        sample_ids = available_ids[:10]
        raise ValueError(
            f"No sensors selected for watershed_id='{watershed_id}'. "
            f"Check watershed IDs in event table and sensor map. Available sensor-map watershed IDs sample: {sample_ids}"
        )
    return selected


def parse_event(
    record: EventRecord,
    sensor_map_df: Optional[pd.DataFrame],
    zip_member_template: Optional[str],
) -> ParsedEvent:
    if not record.source_path.exists():
        raise FileNotFoundError(f"Missing event source file: {record.source_path}")

    frame = parse_hyg_to_dataframe(record.source_path, record.event_id, zip_member_template, record.zip_member_path)
    time_column = detect_time_column(frame)
    time_values = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=np.float64)

    sensor_columns: Dict[int, str] = {}
    for column in frame.columns:
        if column == time_column:
            continue
        loc_id = parse_loc_id(str(column))
        if loc_id is not None:
            sensor_columns[loc_id] = column

    if not sensor_columns:
        raise ValueError(f"No sensor columns detected in {record.source_path}")

    all_sensor_ids = sorted(sensor_columns.keys())
    selected_sensor_ids = select_sensor_ids_for_watershed(all_sensor_ids, record.watershed_id, sensor_map_df)
    selected_columns = [sensor_columns[sensor_id] for sensor_id in selected_sensor_ids]

    values = frame[selected_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid_rows = np.isfinite(time_values)
    valid_rows &= np.all(np.isfinite(values), axis=1)

    time_values = time_values[valid_rows]
    values = values[valid_rows]
    if len(time_values) == 0:
        raise ValueError(f"No valid rows after filtering NaNs for event {record.event_id}")

    return ParsedEvent(
        event_id=record.event_id,
        watershed_id=record.watershed_id,
        time_values=time_values,
        sensor_ids=selected_sensor_ids,
        values=values,
    )


def align_events_trim(events: List[ParsedEvent]) -> List[ParsedEvent]:
    common_t = min(event.values.shape[0] for event in events)
    out: List[ParsedEvent] = []
    for event in events:
        out.append(
            ParsedEvent(
                event_id=event.event_id,
                watershed_id=event.watershed_id,
                time_values=event.time_values[:common_t],
                sensor_ids=event.sensor_ids,
                values=event.values[:common_t],
            )
        )
    LOGGER.info("Applied trim alignment to common T=%d", common_t)
    return out


def align_events_resample(
    events: List[ParsedEvent],
    interval_hours: float,
    drop_first_row: bool,
) -> List[ParsedEvent]:
    if interval_hours <= 0:
        raise ValueError("interval_hours must be > 0 for resampling")

    global_start = max(float(np.min(event.time_values)) for event in events)
    global_end = min(float(np.max(event.time_values)) for event in events)
    if global_end <= global_start:
        raise ValueError("No overlapping time range across events for resampling")

    grid = np.arange(global_start, global_end + interval_hours * 0.5, interval_hours, dtype=np.float64)
    if drop_first_row:
        if grid.shape[0] <= 1:
            raise ValueError("Cannot drop first resampled row because the common grid has length <= 1")
        grid = grid[1:]
    out: List[ParsedEvent] = []
    for event in events:
        interpolated = np.empty((grid.shape[0], event.values.shape[1]), dtype=np.float32)
        for sensor_index in range(event.values.shape[1]):
            interpolated[:, sensor_index] = np.interp(grid, event.time_values, event.values[:, sensor_index]).astype(
                np.float32
            )

        out.append(
            ParsedEvent(
                event_id=event.event_id,
                watershed_id=event.watershed_id,
                time_values=grid,
                sensor_ids=event.sensor_ids,
                values=interpolated,
            )
        )
    LOGGER.info("Applied resample alignment with interval=%.6f hr to common T=%d", interval_hours, grid.shape[0])
    return out


def align_events(
    events: List[ParsedEvent],
    mode: str,
    interval_hours: Optional[float],
    resample_drop_first_row: bool,
) -> List[ParsedEvent]:
    if mode == "trim":
        return align_events_trim(events)
    if mode == "resample":
        if interval_hours is None:
            raise ValueError("resample_interval_hours is required when time_align_mode='resample'")
        return align_events_resample(events, interval_hours, resample_drop_first_row)
    raise ValueError(f"Unsupported time_align_mode: {mode}")


def save_events(events: List[ParsedEvent], output_dir: Path) -> pd.DataFrame:
    events_dir = output_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for event in events:
        event_out_dir = events_dir / event.watershed_id / event.event_id
        event_out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = event_out_dir / "X_event.npy"

        np.save(npy_path, event.values.astype(np.float32, copy=False))
        LOGGER.info(
            "Saved event_id=%s watershed_id=%s shape=(T=%d, F=%d) -> %s",
            event.event_id,
            event.watershed_id,
            event.values.shape[0],
            event.values.shape[1],
            npy_path,
        )

        rows.append(
            {
                "event_id": event.event_id,
                "watershed_id": event.watershed_id,
                "path_to_X_event": str(npy_path),
                "T": int(event.values.shape[0]),
                "F": int(event.values.shape[1]),
            }
        )

    events_df = pd.DataFrame(rows).sort_values(["watershed_id", "event_id"]).reset_index(drop=True)
    events_csv = output_dir / "events.csv"
    events_df.to_csv(events_csv, index=False)
    LOGGER.info("Wrote events table with %d rows: %s", len(events_df), events_csv)
    return events_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 1B: Convert event sources to aligned X_event tensors")
    parser.add_argument("--events-source-csv", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Base dir for relative hyg_path entries")
    parser.add_argument("--zip-member-template", type=str, default=None)
    parser.add_argument("--sensor-map-csv", type=Path, default=None)
    parser.add_argument("--sensor-map-watershed-column", type=str, default="watershed_id")
    parser.add_argument(
        "--sensor-map-default-watershed-id",
        type=str,
        default=None,
        help="Optional override to assign all sensor-map rows to one watershed_id",
    )
    parser.add_argument("--time-align-mode", choices=["trim", "resample"], default="trim")
    parser.add_argument("--resample-interval-hours", type=float, default=None)
    parser.add_argument(
        "--resample-drop-first-row",
        action="store_true",
        help="Drop the first row of the resampled time grid to match the legacy 30-minute preprocessing behavior",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_event_records(args.events_source_csv, args.base_dir)
    LOGGER.info("Loaded %d event records", len(records))

    sensor_map_df = load_sensor_map(
        args.sensor_map_csv,
        args.sensor_map_watershed_column,
        args.sensor_map_default_watershed_id,
    )
    if sensor_map_df is None:
        LOGGER.warning("No sensor map provided: all available Loc* columns will be used per event")
    else:
        LOGGER.info("Loaded sensor map with %d rows", len(sensor_map_df))

    parsed_events: List[ParsedEvent] = []
    for record in records:
        parsed = parse_event(record, sensor_map_df, args.zip_member_template)
        parsed_events.append(parsed)
        LOGGER.info(
            "Parsed event_id=%s watershed_id=%s raw_shape=(T=%d, F=%d)",
            parsed.event_id,
            parsed.watershed_id,
            parsed.values.shape[0],
            parsed.values.shape[1],
        )

    aligned = align_events(
        parsed_events,
        args.time_align_mode,
        args.resample_interval_hours,
        args.resample_drop_first_row,
    )
    save_events(aligned, args.output_dir)


if __name__ == "__main__":
    main()
