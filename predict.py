# predict.py
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import rasterio as rio
from rasterio.merge import merge
import glob
import matplotlib.pyplot as plt
import yaml
import configparser

from data_loader import load_tritonlite_data


# --------------------- Model (must match train.py) ---------------------
class MLP(nn.Module):
    def __init__(self, in_features: int, out_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# --------------------- Utility functions ---------------------
def list_block_paths(base_dir: str, set_code: str, blocks: int, pattern: str):
    paths = []
    for i in range(blocks):
        p = Path(base_dir) / pattern.format(set=set_code, i=i)
        if p.exists(): paths.append((i, p))
    return paths

def write_block_tif(template_path: Path, arr_bhw: np.ndarray, out_path: Path):
    """
    arr_bhw: [bands, H, W] float32
    Uses the template GeoTIFF metadata for CRS, transform, height/width, band count.
    """
    with rio.open(template_path) as src:
        meta = src.meta.copy()
    meta.update(driver="GTiff", count=arr_bhw.shape[0], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_path, "w", **meta) as dst:
        for b in range(arr_bhw.shape[0]):
            dst.write(arr_bhw[b], b + 1)

def create_mosaic(exported_dir: Path, out_path: Path, suffix: str):
    """
    Mosaics all files in exported_dir matching *{suffix}.tif
    """
    tifs = sorted(glob.glob(str(exported_dir / f"*{suffix}")))
    if not tifs:
        print(f"[mosaic] no files found under {exported_dir} with suffix {suffix}")
        return None
    srcs = [rio.open(p) for p in tifs]
    mosaic, out_trans = merge(srcs)
    out_meta = srcs[0].meta.copy()
    out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    for s in srcs: s.close()
    print(f"[mosaic] wrote {out_path}")
    return out_path

def max_over_bands(in_path: Path, out_path: Path):
    with rio.open(in_path) as src:
        meta = src.meta.copy(); meta.update(count=1, dtype=np.float32)
        max_arr = None
        for b in range(1, src.count + 1):
            band = src.read(b)
            max_arr = band if max_arr is None else np.maximum(max_arr, band)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_path, "w", **meta) as dst:
        dst.write(max_arr.astype(np.float32), 1)
    print(f"[max] wrote {out_path}")
    return out_path

# Metrics (vectorized)
def _masks(y_true, y_pred, thr):
    y_true = np.nan_to_num(y_true, nan=thr - thr**2)
    y_pred = np.nan_to_num(y_pred, nan=thr - thr**2)
    t = y_true >= thr; p = y_pred >= thr
    c = t & p; op = (~t) & p; up = t & (~p)
    return dict(c=c.sum(), op=op.sum(), up=up.sum(), n_t=t.sum(), n_p=p.sum())

def f2(y_true, y_pred, thr=0.5):
    m = _masks(y_true, y_pred, thr)
    return (m["c"] - m["op"]) / (m["c"] + m["op"] + m["up"] + 1e-12)

def csi(y_true, y_pred, thr=0.5):
    m = _masks(y_true, y_pred, thr)
    return m["c"] / (m["c"] + m["op"] + m["up"] + 1e-12)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def plot_histogram(diff_path: Path, out_png: Path, bins=5, title="Sugar Creek"):
    with rio.open(diff_path) as src:
        arr = src.read(1)
    data = arr[arr != 0]
    if data.size == 0:
        print("[hist] no nonzero data to plot")
        return
    counts, bin_edges = np.histogram(data, bins=bins)
    percentages = (counts / data.size) * 100.0
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    plt.figure()
    plt.bar(centers, percentages, align='center', width=np.diff(bin_edges), edgecolor='black')
    plt.title(title)
    plt.xlabel('Difference of Water Depth (m)')
    plt.ylabel('Percentage of Cells (%)')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[hist] wrote {out_png}")


# --------------------- Predict pipeline ---------------------
def main():
    # --- Load config (paths, pattern, etc.) ---
    cfg = configparser.ConfigParser()
    cfg.read("Triton_Lite_Ornl/tritonlite_sugar_creek.cfg")

    base_dir           = cfg['Paths']['base_dir']
    hyg_csv            = cfg['Paths']['hyg_dir']
    out_dir_tritonlite = Path(cfg['Paths']['base_dir_tritonlite'])   # predictions
    out_dir_triton     = Path(cfg['Paths']['base_dir_triton'])       # export of GT (optional)
    result_dir         = Path(cfg['Paths']['result_dir'])
    test_set_code      = cfg['Settings']['test_set_code']
    pattern            = "sugar_creek_ACC_{set}_sugar_creek_block_{i}.tif"
    blocks             = int(cfg['block']['block_no'])
    threshold          = float(cfg['Settings'].get('threshold', 0.1))
    columns            = cfg['Columns']['columns_to_keep'].split(',')

    best_yaml        = Path(cfg['Training'].get('best_yaml', 'Triton_Lite_Ornl/artifacts/best_config.yaml'))
    ckpt_path        = Path(cfg['Training'].get('ckpt_path', 'Triton_Lite_Ornl/artifacts/best_config.yaml'))

    # --- Load tuned config + model checkpoint ---
    if not ckpt_path.exists():
        raise FileNotFoundError("artifacts/best.pt not found. Train first with train.py.")
    if best_yaml.exists():
        with open(best_yaml, "r") as f: best_cfg = yaml.safe_load(f)
        hidden  = best_cfg["model"]["hidden"]
        dropout = float(best_cfg["model"]["dropout"])
    else:
        # Fallback defaults (match train.py)
        hidden, dropout = [512, 256], 0.1

    # --- Load data (we only need test set here, but loader returns both) ---
    # Keep these slices consistent with train.py/quick_loader during debugging
    train_sets = [f"D{i:03d}" for i in range(1, 2) if f"D{i:03d}" != "D004"]
    train_slice = slice(0, 480)
    test_slice  = slice(0, None)

    X_tr, X_te, Y_tr, Y_te, meta, _ = load_tritonlite_data(
        base_dir=base_dir,
        hyg_csv=hyg_csv,
        train_sets=train_sets,
        test_set=test_set_code,
        blocks=blocks,
        threshold=threshold,
        columns_to_keep=columns,
        train_row_slice=train_slice,
        test_row_slice=test_slice,
    )
    # Shapes
    in_features = X_te.shape[1]
    out_dim     = Y_te.shape[1]
    bands, H, W = meta["bands"], meta["height"], meta["width"]
    n_blocks    = meta["n_blocks_used"]

    # --- Build and load model ---
    model = MLP(in_features=in_features, out_dim=out_dim, hidden=hidden, dropout=dropout)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Predict on test ---
    X_te_t = torch.from_numpy(X_te)
    with torch.no_grad():
        Y_pred_flat = model(X_te_t).numpy().astype(np.float32)  # [bands, n_blocks*H*W]

    # --- Split predictions into blocks and reshape to [bands, H, W] ---
    # Both Y_pred_flat and Y_te are [bands, n_blocks*H*W]
    def split_and_reshape(Y_flat):
        parts = np.array_split(Y_flat, n_blocks, axis=1)  # list of [bands, H*W]
        return [p.reshape(bands, H, W) for p in parts]    # list of [bands,H,W]

    pred_blocks = split_and_reshape(Y_pred_flat)
    gt_blocks   = split_and_reshape(Y_te)   # optional export of “test” target

    # --- Write block GeoTIFFs using original block metadata as template ---
    test_block_paths = list_block_paths(base_dir, test_set_code, blocks, pattern)
    if len(test_block_paths) != n_blocks:
        print(f"[warn] exporter sees {len(test_block_paths)} existing blocks; loader used {n_blocks}")

    # match by order: we iterate i=0..n_blocks-1
    for i in range(n_blocks):
        # template path: prefer actual file if it exists, else fall back to first available
        tpl = test_block_paths[i][1] if i < len(test_block_paths) else test_block_paths[0][1]
        # pred
        out_pred = out_dir_tritonlite / f"{test_set_code}_block_{i}_pred.tif"
        write_block_tif(tpl, pred_blocks[i], out_pred)
        # gt (optional, for parity with legacy export)
        out_gt = out_dir_triton / f"{test_set_code}_block_{i}_gt.tif"
        write_block_tif(tpl, gt_blocks[i], out_gt)

    print(f"[export] wrote {n_blocks} pred blocks to {out_dir_tritonlite}")
    print(f"[export] wrote {n_blocks} gt blocks to {out_dir_triton}")

    # --- Mosaics ---
    mos_pred = create_mosaic(out_dir_tritonlite, out_dir_tritonlite / "mosaic" / "mosaic_tritonlite_pred.tif", suffix="_pred.tif")
    mos_gt   = create_mosaic(out_dir_triton,     out_dir_triton     / "mosaic" / "mosaic_triton_gt.tif",       suffix="_gt.tif")

    # --- Max over bands (MOM) ---
    if mos_pred is None or mos_gt is None:
        print("[skip] mosaics missing; cannot compute MOM and diff.")
        return
    mom_pred = max_over_bands(mos_pred, out_dir_tritonlite / "mosaic" / "MOM_tritonlite_pred.tif")
    mom_gt   = max_over_bands(mos_gt,   out_dir_triton     / "mosaic" / "MOM_triton_gt.tif")

    # --- Subtraction (GT - PRED) and metrics ---
    diff_path = result_dir / "diff_pred_vs_gt.tif"
    with rio.open(mom_gt) as src1, rio.open(mom_pred) as src2:
        a = src1.read(1, masked=True).filled(0).astype(np.float32)
        b = src2.read(1, masked=True).filled(0).astype(np.float32)
        Hm, Wm = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a = a[:Hm, :Wm]; b = b[:Hm, :Wm]
        diff = a - b
        meta_out = src1.meta.copy(); meta_out.update(count=1, height=Hm, width=Wm, dtype=np.float32)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(diff_path, "w", **meta_out) as dst:
        dst.write(diff.astype(np.float32), 1)
    print(f"[diff] wrote {diff_path}")

    # Metrics at threshold 0.1 (match your legacy code)
    thr = 0.1
    print(f"F2  : {f2(a, b, thr):.6f}")
    print(f"CSI : {csi(a, b, thr):.6f}")
    print(f"RMSE: {rmse(a, b):.6f}")

    # --- Histogram plot of differences ---
    plot_histogram(diff_path, result_dir / "sugar_creek_histogram.png", bins=5, title="Sugar Creek")


if __name__ == "__main__":
    main()
