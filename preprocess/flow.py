# flow_process.py
import os, io, re, zipfile, argparse, configparser
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- helpers ----------------
def _process_txt_bytes(txt_bytes: bytes, n_locations: int = 300, comment_prefix: str = "%") -> pd.DataFrame:
    """Parse hyg text (bytes) → DataFrame with ['Time (hr)', Loc1..LocN]."""
    text = txt_bytes.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(comment_prefix)]
    if not lines:
        raise ValueError("No data lines after removing comments.")
    buf = io.StringIO("\n".join(lines))

    # try whitespace-delimited, then comma-delimited
    df = None
    for kwargs in (dict(delim_whitespace=True, header=None), dict(sep=",", header=None)):
        buf.seek(0)
        try:
            df = pd.read_csv(buf, **kwargs)
            break
        except Exception:
            df = None
    if df is None:
        raise ValueError("Unable to parse HYG text as whitespace or CSV.")

    # pad if not enough columns
    expect_cols = n_locations + 1
    if df.shape[1] < expect_cols:
        for _ in range(expect_cols - df.shape[1]):
            df[f"_pad_{_}"] = np.nan
        df = df.iloc[:, :expect_cols]

    cols = ["Time (hr)"] + [f"Loc{i}" for i in range(1, n_locations + 1)]
    df.columns = cols
    # coerce numeric & sort by time
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates(subset=["Time (hr)"]).sort_values("Time (hr)").reset_index(drop=True)
    return df


def _find_hyg_member(zf: zipfile.ZipFile, pattern=r".*/input/hyg/.*\.txt") -> str | None:
    rx = re.compile(pattern)
    for name in zf.namelist():
        if rx.fullmatch(name) or rx.match(name):
            return name
    return None


# ---------------- step 1: extract ----------------
def extract_from_zips(cfg: configparser.ConfigParser, n_locations: int = 300) -> None:
    """
    Reads D###.zip from zip_dir (or input_dir if zip_dir missing),
    writes processed_data_D###.csv into input_dir (so step 2 can use them).
    """
    S = cfg["Settings"]
    input_dir  = Path(S["input_dir"])          # where we will write processed_data_D###.csv
    zip_dir    = Path(S.get("zip_dir", S["input_dir"]))  # default: same as input_dir
    start_ev   = int(S["start_event"])
    end_ev     = int(S["end_event"])

    input_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start_ev, end_ev + 1):
        ev = f"D{i:03d}"
        zip_path = zip_dir / f"{ev}.zip"
        out_csv  = input_dir / f"processed_data_{ev}.csv"

        if out_csv.exists():
            print(f"[extract] exists, skip: {out_csv}")
            continue
        if not zip_path.exists():
            print(f"[extract] MISSING zip: {zip_path}")
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                member = _find_hyg_member(zf)
                if member is None:
                    print(f"[extract] No */input/hyg/*.txt inside {zip_path}")
                    continue
                with zf.open(member) as f:
                    df = _process_txt_bytes(f.read(), n_locations=n_locations)
            df.to_csv(out_csv, index=False)
            print(f"[extract] wrote {out_csv} (rows={len(df)}, cols={df.shape[1]})")
        except Exception as e:
            print(f"[extract] ERROR {ev}: {e}")


# ---------------- step 2: resample ----------------
def resample_to_interval(cfg: configparser.ConfigParser) -> None:
    """
    Reads processed_data_D###.csv from input_dir,
    writes interpolated_data_processed_data_D###.csv to output_dir.
    """
    S = cfg["Settings"]
    input_dir   = Path(S["input_dir"])
    output_dir  = Path(S["output_dir"])
    new_dt      = float(S["new_interval"])     # hours (e.g., 0.5 = 30 min)
    start_ev    = int(S["start_event"])
    end_ev      = int(S["end_event"])

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start_ev, end_ev + 1):
        ev = f"D{i:03d}"
        inp = input_dir  / f"processed_data_{ev}.csv"
        out = output_dir / f"interpolated_data_processed_data_{ev}.csv"

        if out.exists():
            print(f"[resample] exists, skip: {out}")
            continue
        if not inp.exists():
            print(f"[resample] missing input, skip: {inp}")
            continue

        df = pd.read_csv(inp)
        if "Time (hr)" not in df.columns:
            print(f"[resample] invalid schema (no 'Time (hr)'): {inp}")
            continue

        t = pd.to_numeric(df["Time (hr)"], errors="coerce").to_numpy()
        mask_t = np.isfinite(t)
        t = t[mask_t]
        if t.size < 2:
            print(f"[resample] not enough time points in {inp}")
            continue

        t_min, t_max = float(np.min(t)), float(np.max(t))
        new_t = np.arange(t_min, t_max + 1e-9, new_dt, dtype=float)

        out_df = pd.DataFrame({"Time (hr)": new_t})
        for col in df.columns:
            if col == "Time (hr)":
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy()
            y = y[: len(mask_t)]
            finite = mask_t & np.isfinite(y)
            if finite.sum() < 2:
                out_df[col] = np.nan
                continue
            out_df[col] = np.interp(new_t, t[finite], y[finite])

        # Match your original: drop the first row after interpolation
        if len(out_df) > 0:
            out_df = out_df.iloc[1:].reset_index(drop=True)

        out_df.to_csv(out, index=False)
        print(f"[resample] wrote {out} (rows={len(out_df)}, cols={out_df.shape[1]})")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser("Flow data processing (extract + resample)")
    ap.add_argument("--cfg", required=True, help="Path to .cfg (with [Settings])")
    ap.add_argument("--step", choices=["extract", "resample", "both"], default="both")
    ap.add_argument("--n_locations", type=int, default=300, help="Number of Loc columns to expect")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.cfg)

    if args.step in ("extract", "both"):
        extract_from_zips(cfg, n_locations=args.n_locations)
    if args.step in ("resample", "both"):
        resample_to_interval(cfg)


if __name__ == "__main__":
    main()

'''
# Extract hydrographs to processed_data_D###.csv (written into input_dir)
python flow_process.py --cfg convert_hyg_3hrs_to_30mins.cfg --step extract

# Resample to new interval, write to output_dir
python flow_process.py --cfg convert_hyg_3hrs_to_30mins.cfg --step resample

# run both steps
python flow_process.py --cfg convert_hyg_3hrs_to_30mins.cfg --step both
'''