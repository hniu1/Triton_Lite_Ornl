# train.py
import os
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_loader import load_tritonlite_data
import configparser


# --------------------- utils ---------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class MLP(nn.Module):
    """Simple MLP: [B, F] -> [B, K]. Mirrors Conv1D(k=1) capacity."""
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


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


# --------------------- data ---------------------
def get_data_from_cfg(cfg_path: str):
    config = configparser.ConfigParser()
    config.read(cfg_path)

    base_dir = config['Paths']['base_dir']
    hyg_csv  = config['Paths']['hyg_dir']
    blocks   = int(config['block']['block_no'])
    threshold = float(config['Settings'].get('threshold', 0.1))
    columns = config['Columns']['columns_to_keep'].split(',')

    # Keep small for CPU sanity runs; expand later.
    train_sets = [f"D{i:03d}" for i in range(1, 2) if f"D{i:03d}" != "D004"]
    test_set   = "D004"

    # These slices should match the ones you tested in quick_loader
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
    return X_tr, Y_tr, X_te, Y_te, meta


# --------------------- main train ---------------------
def main():
    set_seed(42)
    Path("artifacts").mkdir(exist_ok=True)

    # 1) Load data
    cfg_path = "Triton_Lite_Ornl/tritonlite_sugar_creek.cfg"
    X_tr, Y_tr, X_te, Y_te, meta = get_data_from_cfg(cfg_path)
    print(f"[Data] X_tr={X_tr.shape}, Y_tr={Y_tr.shape} | X_te={X_te.shape}, Y_te={Y_te.shape}")

    # 2) Load tuned config if present, else defaults
    best_cfg_path = Path("Triton_Lite_Ornl/artifacts/best_config.yaml")
    if best_cfg_path.exists():
        with open(best_cfg_path, "r") as f:
            best_cfg = yaml.safe_load(f)
        hidden   = best_cfg["model"]["hidden"]
        dropout  = float(best_cfg["model"]["dropout"])
        lr       = float(best_cfg["train"]["lr"])
        batch_sz = int(best_cfg["train"]["batch_size"])
        max_epochs = int(best_cfg["train"].get("epochs", 100))
        val_ratio = float(best_cfg["train"].get("val_ratio", 0.1))
        es_patience = int(best_cfg["train"].get("early_stop_patience", 10))
        print("[Config] Loaded tuned params from artifacts/best_config.yaml")
    else:
        hidden   = [512, 256]
        dropout  = 0.1
        lr       = 1e-3
        batch_sz = 128
        max_epochs = 100
        val_ratio = 0.1
        es_patience = 10
        print("[Config] Using default training params")

    # 3) Torch tensors + split
    X_trn, X_val, Y_trn, Y_val = train_test_split(X_tr, Y_tr, test_size=val_ratio, random_state=42, shuffle=True)
    X_trn_t = torch.from_numpy(X_trn);  Y_trn_t = torch.from_numpy(Y_trn)
    X_val_t = torch.from_numpy(X_val);  Y_val_t = torch.from_numpy(Y_val)
    X_te_t  = torch.from_numpy(X_te);   Y_te_t  = torch.from_numpy(Y_te)

    dl_trn = DataLoader(TensorDataset(X_trn_t, Y_trn_t), batch_size=batch_sz, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_sz, shuffle=False)
    dl_te  = DataLoader(TensorDataset(X_te_t,  Y_te_t),  batch_size=batch_sz, shuffle=False)

    # 4) Model/optim
    in_features = X_tr.shape[1]
    out_dim = Y_tr.shape[1]
    model = MLP(in_features=in_features, out_dim=out_dim, hidden=hidden, dropout=dropout)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Train with early stopping
    best_val = float("inf"); patience = 0
    ckpt_path = "Triton_Lite_Ornl/artifacts/best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl_trn:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validation
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= max(1, len(dl_val))
        print(f"epoch {epoch:03d}  val_mse={val_loss:.6f}")

        # early stop + checkpoint
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            patience = 0
            torch.save({"model": model.state_dict(),
                        "in_features": in_features,
                        "out_dim": out_dim,
                        "hidden": hidden,
                        "dropout": dropout}, ckpt_path)
        else:
            patience += 1
            if patience >= es_patience:
                print("Early stopping.")
                break

    # 6) Evaluate on test set using the best checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_mse = 0.0
    with torch.no_grad():
        for xb, yb in dl_te:
            test_mse += loss_fn(model(xb), yb).item()
    test_mse /= max(1, len(dl_te))
    print(f"[Test] MSE = {test_mse:.6f}")

    # You now have a trained model checkpoint in artifacts/best.pt
    # Next steps (later): reshape predictions back to [bands, H, W] per block and export GeoTIFFs.

if __name__ == "__main__":
    main()
