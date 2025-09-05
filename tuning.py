# tune.py
import os
from pathlib import Path
import random, numpy as np, yaml
import optuna
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- import your working loader ---
from data_loader import load_tritonlite_data
import configparser


# --------------------- utils ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """
    Equivalent capacity to Conv1D with kernel_size=1 over a length-1 axis:
    just a stack of Linear + ReLU (+ Dropout), mapping [B, F] -> [B, K].
    """
    def __init__(self, in_features: int, out_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------- data (calls your loader) ---------------------
def get_data_from_cfg(cfg_path: str):
    """
    Reads the same .cfg you already used and calls load_tritonlite_data.
    Adjust only the TRAIN/TEST slices and blocks here for quick CPU tests.
    """
    config = configparser.ConfigParser()
    config.read(cfg_path)

    base_dir = config['Paths']['base_dir']
    hyg_csv  = config['Paths']['hyg_dir']
    blocks   = int(config['block']['block_no'])
    threshold = float(config['Settings'].get('threshold', 0.1))
    columns = config['Columns']['columns_to_keep'].split(',')

    # train/test sets (example: train D001..D003 except D004, test D004)
    # Adjust if you want more sets included.
    train_sets = [f"D{i:03d}" for i in range(1, 2) if f"D{i:03d}" != "D004"]  # keep small for CPU tuning
    test_set   = "D004"

    # row slices — match what you used in your successful loader test
    train_slice = slice(0, 480)
    test_slice  = slice(0, None)

    X_tr, X_te, Y_tr, Y_te, meta, scaler = load_tritonlite_data(
        base_dir=base_dir,
        hyg_csv=hyg_csv,
        train_sets=train_sets,
        test_set=test_set,
        blocks=blocks,
        threshold=threshold,
        columns_to_keep=columns,
        train_row_slice=train_slice,
        test_row_slice=test_slice,
    )
    return X_tr, Y_tr, X_te, Y_te, meta, scaler


# --------------------- objective ---------------------
def objective(trial: optuna.Trial, X_tr: np.ndarray, Y_tr: np.ndarray, val_ratio: float = 0.1):
    set_seed(42)

    # Search space (roughly mirrors your Keras tuner choices)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    h1 = trial.suggest_int("hidden1", 16, 1024, step=16)
    h2 = trial.suggest_int("hidden2", 16, 512, step=16)
    h3 = trial.suggest_int("hidden3", 0, 1024, step=32)  # 0 means skip third layer
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    hidden = [h1, h2] + ([h3] if h3 > 0 else [])

    # Split train -> train/val (no leakage)
    X_trn, X_val, Y_trn, Y_val = train_test_split(
        X_tr, Y_tr, test_size=val_ratio, random_state=42, shuffle=True
    )

    # Torch tensors + loaders
    X_trn_t = torch.from_numpy(X_trn)
    Y_trn_t = torch.from_numpy(Y_trn)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    dl_trn = DataLoader(TensorDataset(X_trn_t, Y_trn_t), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)

    in_features = X_tr.shape[1]
    out_dim = Y_tr.shape[1]

    model = MLP(in_features=in_features, out_dim=out_dim, hidden=hidden, dropout=dropout)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    patience = 0
    max_epochs = 40  # short trials; we’ll early stop
    es_patience = 5

    for epoch in range(max_epochs):
        # train
        model.train()
        for xb, yb in dl_trn:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
        val_loss /= max(1, len(dl_val))

        # report to Optuna + pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= es_patience:
                break

    return best_val


# --------------------- main ---------------------
def main():
    # 1) Load data through your loader (no model involved)
    cfg_path = "Triton_Lite_Ornl/tritonlite_sugar_creek.cfg"
    X_tr, Y_tr, X_te, Y_te, meta, scaler = get_data_from_cfg(cfg_path)

    print(f"[Data] X_tr={X_tr.shape}, Y_tr={Y_tr.shape} | X_te={X_te.shape}, Y_te={Y_te.shape}")
    print(f"[Meta] bands={meta['bands']} H={meta['height']} W={meta['width']} blocks_used={meta['n_blocks_used']}")

    # 2) Optuna study (median pruner ~ your random search w/ early stopping)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=optuna.samplers.RandomSampler())
    study.optimize(lambda t: objective(t, X_tr, Y_tr, val_ratio=0.10), n_trials=20)

    best = study.best_trial
    print(f"\nBest val MSE: {best.value:.6f}")
    print("Best params:", best.params)

    # 3) Save a simple config you can reuse in train.py
    out = {
        "model": {
            "name": "mlp",
            "hidden": [best.params["hidden1"], best.params["hidden2"]] + ([best.params["hidden3"]] if best.params["hidden3"] > 0 else []),
            "dropout": best.params["dropout"],
            "out_dim": int(Y_tr.shape[1]),
        },
        "train": {
            "lr": float(best.params["lr"]),
            "batch_size": int(best.params["batch_size"]),
            "epochs": 100,
            "val_ratio": 0.10,
            "early_stop_patience": 10,
        },
        "data": {
            "in_features": int(X_tr.shape[1]),
            "bands": int(meta["bands"]),
            "height": int(meta["height"]),
            "width": int(meta["width"]),
        }
    }
    Path("Triton_Lite_Ornl/artifacts").mkdir(exist_ok=True)
    with open("Triton_Lite_Ornl/artifacts/best_config.yaml", "w") as f:
        yaml.safe_dump(out, f)
    print("Saved best config -> artifacts/best_config.yaml")


if __name__ == "__main__":
    set_seed(42)
    main()
