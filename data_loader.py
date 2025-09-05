# quick_loader.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.preprocessing import StandardScaler
import configparser

def _load_set_stack(
    base_dir: str,
    set_code: str,
    blocks: int,
    threshold: float,
    pattern: str = "sugar_creek_ACC_{set}_sugar_creek_block_{i}.tif",
) -> Tuple[np.ndarray, Dict]:
    """Load one set -> hstack blocks -> threshold -> [bands, blocks*H*W]."""
    base = Path(base_dir)
    per_block = []
    bands = h = w = None
    for i in range(blocks):
        fp = base / pattern.format(set=set_code, i=i)
        if not fp.exists():
            # skip missing blocks
            continue
        with rio.open(fp) as src:
            if bands is None:
                bands, h, w = src.count, src.height, src.width
            arr = src.read().astype(np.float32)        # [B,H,W]
            per_block.append(arr.reshape(arr.shape[0], -1))  # [B,H*W]
    if not per_block:
        raise RuntimeError(f"No valid blocks found for {set_code}")
    Y = np.hstack(per_block)                           # [B, blocks*H*W]
    Y[Y < threshold] = 0.0
    meta = {"bands": bands, "height": h, "width": w, "n_blocks_used": len(per_block)}
    return Y, meta

def load_tritonlite_data(
    base_dir: str,
    hyg_csv: str,
    train_sets: List[str],
    test_set: str,
    *,
    blocks: int,
    threshold: float = 0.1,
    columns_to_keep: List[str] = None,
    train_row_slice: slice = slice(0, 18240),
    test_row_slice: slice = slice(18240, None),
    pattern: str = "sugar_creek_ACC_{set}_sugar_creek_block_{i}.tif",
):
    """
    Returns:
      X_train [N_train, F], X_test [N_test, F],
      Y_train_flat [N_train, K], Y_test_flat [N_test, K],
      meta (bands/height/width), scaler (fitted StandardScaler)
    """
    # --- Targets (rasters) ---
    # Train: load each set, then vstack across sets along sample axis
    Y_train_list = []
    shared_meta = None
    for s in train_sets:
        Y_set, meta = _load_set_stack(base_dir, s, blocks, threshold, pattern)
        if shared_meta is None:
            shared_meta = meta
        else:
            # sanity: consistent band/shape
            assert (meta["bands"], meta["height"], meta["width"]) == \
                   (shared_meta["bands"], shared_meta["height"], shared_meta["width"]), \
                   f"Shape mismatch in set {s}"
        Y_train_list.append(Y_set)
    if not Y_train_list:
        raise RuntimeError("No training data found.")
    Y_train = np.vstack(Y_train_list)                  # [sum_sets*bands, blocks*H*W]?  (matches your script)

    # Test: single set
    Y_test, _ = _load_set_stack(base_dir, test_set, blocks, threshold, pattern)

    # Flatten targets to [N, K] for MLPs
    Y_train_flat = Y_train.reshape(Y_train.shape[0], -1).astype(np.float32)
    Y_test_flat  = Y_test.reshape(Y_test.shape[0], -1).astype(np.float32)

    # --- Inputs (tabular X) ---
    if columns_to_keep is None:
        raise ValueError("columns_to_keep must be provided.")
    df = pd.read_csv(hyg_csv, usecols=columns_to_keep)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.iloc[train_row_slice].to_numpy()).astype(np.float32)
    X_test  = scaler.transform(df.iloc[test_row_slice].to_numpy()).astype(np.float32)

    # sanity: row counts match between X and Y
    assert X_train.shape[0] == Y_train_flat.shape[0], f"X_train rows {X_train.shape[0]} != Y_train rows {Y_train_flat.shape[0]}"
    assert X_test.shape[0]  == Y_test_flat.shape[0],  f"X_test rows {X_test.shape[0]}  != Y_test rows {Y_test_flat.shape[0]}"

    return X_train, X_test, Y_train_flat, Y_test_flat, shared_meta, scaler


# # Load configuration from config.cfg
# config = configparser.ConfigParser()
# config.read('Triton_Lite_Ornl/tritonlite_sugar_creek.cfg')

# # Extract variables from config
# columns = config['Columns']['columns_to_keep'].split(',')

# # columns = ['Time (hr)'] + [f'Loc{i}' for i in range(146, 156)]
# train_sets = [f"D{i:03d}" for i in range(1, 2) if f"D{i:03d}" != "D004"]
# test_set = "D004"

# X_tr, X_te, Y_tr, Y_te, meta, scaler = load_tritonlite_data(
#     base_dir="Shared_from_Sudershan/data/Block_tiffs",
#     hyg_csv="Shared_from_Sudershan/data/Processed_30min/interpolated_data_processed_data_D001.csv",
#     train_sets=train_sets,
#     test_set=test_set,
#     blocks=5,
#     threshold=0.1,
#     columns_to_keep=columns,
#     train_row_slice=slice(0, 480),
#     test_row_slice=slice(0, None),
# )

# print("X_tr:", X_tr.shape, "Y_tr:", Y_tr.shape)
# print("X_te:", X_te.shape, "Y_te:", Y_te.shape)
# print("meta:", meta)