# flow_process.py
import os, io, re, zipfile, argparse, configparser
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- helpers ----------------

def process_txt_content(txt_content):
    csv_content = '\n'.join(line for line in txt_content.splitlines() if not line.startswith('%'))
    df = pd.read_csv(io.StringIO(csv_content), header=None)
    df.columns = ['Time (hr)'] + [f'Loc{i}' for i in range(1, 301)]
    return df

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

        txt_file_path = f'D{i:03}/input/hyg/D001.txt'

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(txt_file_path) as file:
                txt_content = file.read().decode('utf-8')
                df = process_txt_content(txt_content)


        # Check if the file already exists and is not in use
        
        df.to_csv(out_csv, index=False)
        print(f'Data from {zip_path} processed and saved to {out_csv}')
        

# ---------------- step 2: resample ----------------
def resample_to_interval(cfg: configparser.ConfigParser) -> None:
    """
    Reads processed_data_D###.csv from input_dir,
    writes interpolated_data_processed_data_D###.csv to output_dir.
    """
    S = cfg["Settings"]
    input_dir   = Path(S["input_dir"])
    output_dir  = Path(S["output_dir"])
    new_interval      = float(S["new_interval"])     # hours (e.g., 0.5 = 30 min)
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

        data = pd.read_csv(inp)
        # Create a new DataFrame for the interpolated time values
        new_time_values = np.arange(0, data['Time (hr)'].iloc[-1] + new_interval, new_interval)
        interpolated_data = pd.DataFrame(new_time_values, columns=['Time (hr)'])

        # Interpolate the data for each location
        for location in data.columns[1:]:
            interpolated_data = interpolated_data.join(
                pd.DataFrame(
                    np.interp(
                        interpolated_data['Time (hr)'],
                        data['Time (hr)'],
                        data[location]
                    ),
                    columns=[location]
                )
            )

        # Delete the first row of the interpolated data
        interpolated_data = interpolated_data.drop(interpolated_data.index[0])

        # Save the interpolated data to a new CSV file
        interpolated_data.to_csv(out, index=False)
        print(f'Interpolated file saved to {out}')


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser("Flow data processing (extract + resample)")
    ap.add_argument("--cfg", default="/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/preprocess/convert_hyg_3hrs_to_30mins.cfg", help="Path to .cfg (with [Settings])")
    ap.add_argument("--step", choices=["extract", "resample", "both"], default="both")
    ap.add_argument("--n_locations", type=int, default=300, help="Number of Loc columns to expect")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.cfg)

    # if args.step in ("extract", "both"):
    #     extract_from_zips(cfg, n_locations=args.n_locations)
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