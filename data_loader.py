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
    pattern: str,
    watershed_name_lower: str
) -> Tuple[np.ndarray, Dict]:
    """Load one set -> hstack blocks -> threshold -> [bands, blocks*H*W]."""
    base = Path(base_dir)
    per_block = []
    bands = h = w = None
    for i in range(blocks):
        fp = base / pattern.format(watershed_name_lower=watershed_name_lower, set=set_code, i=i)
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
    hyg_dir: str,
    train_sets: List[str],
    test_set: str,
    # *,
    blocks: int,
    threshold: float = 0.1,
    columns_to_keep: List[str] = None,
    # train_row_slice: slice = slice(0, 18240),
    # test_row_slice: slice = slice(18240, None),
    watershed_name_lower: str = "unknown_watershed",
    pattern: str = "{watershed_name_lower}_ACC_{set}_{watershed_name_lower}_block_{i}.tif",
):
    """
    Returns:
      x_train [N_train, steps, features],
      x_test  [N_test,  steps, features],
      Y_train_flat [N_train, K],
      Y_test_flat  [N_test,  K],
      meta (bands/height/width),
      scaler (fitted StandardScaler)
    """
    # --- Targets (rasters) ---
    Y_train_list = []
    shared_meta = None
    for s in train_sets:
        Y_set, meta = _load_set_stack(base_dir, s, blocks, threshold, pattern, watershed_name_lower)
        if shared_meta is None:
            shared_meta = meta
        else:
            assert (meta["bands"], meta["height"], meta["width"]) == \
                   (shared_meta["bands"], shared_meta["height"], shared_meta["width"]), \
                   f"Shape mismatch in set {s}"
        Y_train_list.append(Y_set)

    if not Y_train_list:
        raise RuntimeError("No training data found.")

    Y_train = np.vstack(Y_train_list)
    Y_test, _ = _load_set_stack(base_dir, test_set, blocks, threshold, pattern, watershed_name_lower)

    # Flatten targets
    Y_train_flat = Y_train.reshape(Y_train.shape[0], -1).astype(np.float32)
    Y_test_flat  = Y_test.reshape(Y_test.shape[0], -1).astype(np.float32)

    # --- Inputs (tabular X) ---
    if columns_to_keep is None:
        raise ValueError("columns_to_keep must be provided.")

    # Load HYG CSVs in train_sets and test_sets separately under hyg_dir
    def load_hyg_files(sets: List[str]) -> pd.DataFrame:
        files = []
        for s in sets:
            files.extend(Path(hyg_dir).glob(f"interpolated_data_processed_data_{s}.csv"))
        if not files:
            raise RuntimeError(f"No HYG CSV files found for sets: {sets}")
        df_list = []
        for f in files:
            df_list.append(pd.read_csv(f, usecols=columns_to_keep))
        return pd.concat(df_list, ignore_index=True)
    
    df_train = load_hyg_files(train_sets)
    df_test  = load_hyg_files([test_set])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.to_numpy()).astype(np.float32)
    X_test  = scaler.transform(df_test.to_numpy()).astype(np.float32)

    # Sanity checks
    assert X_train.shape[0] == Y_train_flat.shape[0], \
        f"X_train rows {X_train.shape[0]} != Y_train rows {Y_train_flat.shape[0]}"
    assert X_test.shape[0] == Y_test_flat.shape[0], \
        f"X_test rows {X_test.shape[0]} != Y_test rows {Y_test_flat.shape[0]}"

    # --- Reshape for Conv1D input ---
    # Keras-style (steps, features) → here steps=1
    x_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    x_test  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    steps = x_train.shape[1]
    features = x_train.shape[2]

    print(f"Prepared data for Conv1D: x_train {x_train.shape}, x_test {x_test.shape}, steps={steps}, features={features}")

    return x_train, x_test, Y_train_flat, Y_test_flat, shared_meta, scaler



# --------------------- data (calls your loader) ---------------------
def get_data_from_cfg(cfg_path: str, 
                      train_sets: List[str], 
                      test_set: str):
    """
    Reads the same .cfg you already used and calls load_tritonlite_data.
    Adjust only the TRAIN/TEST slices and blocks here for quick CPU tests.
    """
    config = configparser.ConfigParser()
    config.read(cfg_path)

    watershed_name = config['Paths'].get('watershed_name', 'unknown_watershed')
    if watershed_name == 'unknown_watershed':
        raise ValueError("watershed_name must be specified in the config file under [Paths].")
    else:
        print(f"Using watershed: {watershed_name}")
    watershed_name_lower = watershed_name.replace(" ", "_").lower()
    ws_dir = config['Paths']['workspace_dir']
    base_dir = Path(f"{ws_dir}/{config['Paths']['base_dir']}_{watershed_name_lower}")
    hyg_dir  = Path(f"{ws_dir}/{config['Paths']['hyg_dir']}")
    blocks   = int(config['block']['block_no'])
    threshold = float(config['Settings'].get('threshold', 0.1))
    columns = config['Columns']['columns_to_keep'].split(',') 

    X_tr, X_te, Y_tr, Y_te, meta, scaler = load_tritonlite_data(
        base_dir=base_dir,
        hyg_dir=hyg_dir,
        train_sets=train_sets,
        test_set=test_set,
        blocks=blocks,
        threshold=threshold,
        columns_to_keep=columns,
        # train_row_slice=train_slice,
        # test_row_slice=test_slice,
        watershed_name_lower=watershed_name_lower
    )
    return X_tr, Y_tr, X_te, Y_te, meta, scaler