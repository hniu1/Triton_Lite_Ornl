# tune.py
from pathlib import Path
import random
import numpy as np
import yaml
import optuna
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- import your working loader ---
from data_loader import get_data_from_cfg
from model import TritonCNN

# --------------------- utils ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------- objective ---------------------
def objective(trial: optuna.Trial, X_tr: np.ndarray, Y_tr: np.ndarray, val_ratio: float = 0.1):
    set_seed(42)

    # --- Search space (mirrors your old Keras tuner) ---
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    conv1_filters = trial.suggest_categorical("conv1_filters", [16, 32, 64])
    conv2_filters = trial.suggest_categorical("conv2_filters", [64, 128, 256])
    dense1_units = trial.suggest_categorical("dense1_units", [16, 32, 64])
    dense2_units = trial.suggest_categorical("dense2_units", [128, 256, 512])
    dense3_units = trial.suggest_categorical("dense3_units", [256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 10])
    dropout    = trial.suggest_float("dropout", 0.0, 0.3)

    # --- Data split ---
    X_trn, X_val, Y_trn, Y_val = train_test_split(X_tr, Y_tr, test_size=val_ratio, random_state=42, shuffle=True)
    X_trn_t, Y_trn_t = torch.from_numpy(X_trn).float(), torch.from_numpy(Y_trn).float()
    X_val_t, Y_val_t = torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()

    dl_trn = DataLoader(TensorDataset(X_trn_t, Y_trn_t), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)

    # --- Model definition (CNN analog of create_cnn_model) ---
    in_features = X_tr.shape[2] if X_tr.ndim == 3 else X_tr.shape[1]
    out_dim = Y_tr.shape[1]

    model = TritonCNN(
        in_features=in_features,
        out_dim=out_dim,
        conv1_filters=conv1_filters,
        conv2_filters=conv2_filters,
        dense1_units=dense1_units,
        dense2_units=dense2_units,
        dense3_units=dense3_units,
        dropout=dropout
    )

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    patience = 0
    es_patience = 5
    max_epochs = 10

    # --- Training loop ---
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in dl_trn:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
        val_loss /= max(1, len(dl_val))

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val - 1e-6:
            best_val, patience = val_loss, 0
        else:
            patience += 1
            if patience >= es_patience:
                break

    return best_val


# --------------------- main ---------------------
def main():
    # 1) Load data through your loader (no model involved)
    cfg_path = Path(f"{path_ws}/tritonlite_sugar_creek.cfg")
    test_set   = "D004"
    train_sets = [f"D{i:03d}" for i in range(1, 4) if f"D{i:03d}" != test_set]  # keep small for CPU tuning

    X_tr, Y_tr, X_te, Y_te, meta, scaler = get_data_from_cfg(cfg_path, train_sets=train_sets, test_set=test_set)

    print(f"[Data] X_tr={X_tr.shape}, Y_tr={Y_tr.shape} | X_te={X_te.shape}, Y_te={Y_te.shape}")
    print(f"[Meta] bands={meta['bands']} H={meta['height']} W={meta['width']} blocks_used={meta['n_blocks_used']}")

    # 2) Optuna study (median pruner ~ your random search w/ early stopping)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=optuna.samplers.RandomSampler())
    study.optimize(lambda t: objective(t, X_tr, Y_tr, val_ratio=0.10), n_trials=2)

    best = study.best_trial
    print(f"\nBest val MSE: {best.value:.6f}")
    print("Best params:", best.params)

    # 3) Save a simple config you can reuse in train.py
    out = {
        "model": {
            "name": "cnn",
            "conv1_filters": int(best.params["conv1_filters"]),
            "conv2_filters": int(best.params["conv2_filters"]),
            "dense1_units": int(best.params["dense1_units"]),
            "dense2_units": int(best.params["dense2_units"]),
            "dense3_units": int(best.params["dense3_units"]),
            "dropout": float(best.params.get("dropout", 0.0)),   # include if you tune dropout
            "out_dim": int(Y_tr.shape[1]),
        },
        "train": {
            "lr": float(best.params["lr"]),
            "batch_size": int(best.params["batch_size"]),
            "epochs": 20,
            "val_ratio": 0.10,
            "early_stop_patience": 10,
        },
        "data": {
            # CNN input has shape (batch, steps, features)
            "steps": 1,
            "features": int(X_tr.shape[-1]),
            "bands": int(meta["bands"]),
            "height": int(meta["height"]),
            "width": int(meta["width"]),
        },
    }

    # creat artifacts directory under path_ws
    Path(f"{path_ws}/results/artifacts").mkdir(exist_ok=True)
    with open(Path(f"{path_ws}/results/artifacts/best_config.yaml"), "w") as f:
        yaml.safe_dump(out, f)
    print("Saved best config -> artifacts/best_config.yaml")


if __name__ == "__main__":
    set_seed(42)
    path_ws = Path("/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl")
    main()
