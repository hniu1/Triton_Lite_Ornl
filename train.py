# train.py
import configparser
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_loader import get_data_from_cfg
from model import TritonCNN

# --------------------- utils ---------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


# --------------------- main train ---------------------
def main():
    set_seed(42)
    path_results = Path(f"{path_ws}/results_{watershed_name_lower}")
    path_results.mkdir(exist_ok=True)
    Path(f"{path_results}/artifacts").mkdir(exist_ok=True)

    # 1) Load data
    
    X_tr, Y_tr, X_te, Y_te, _, _ = get_data_from_cfg(cfg_path, train_sets=train_sets, test_set=test_set)
    print(f"[Data] X_tr={X_tr.shape}, Y_tr={Y_tr.shape} | X_te={X_te.shape}, Y_te={Y_te.shape}")

    # 2) Load tuned config if present, else defaults
    best_cfg_path = Path(f"{path_results}/artifacts/best_config.yaml")
    if best_cfg_path.exists():
        with open(best_cfg_path, "r") as f:
            best_cfg = yaml.safe_load(f)

        # --- Model hyperparameters ---
        conv1_filters = int(best_cfg["model"]["conv1_filters"])
        conv2_filters = int(best_cfg["model"]["conv2_filters"])
        dense1_units  = int(best_cfg["model"]["dense1_units"])
        dense2_units  = int(best_cfg["model"]["dense2_units"])
        dense3_units  = int(best_cfg["model"]["dense3_units"])
        dropout       = float(best_cfg["model"].get("dropout", 0.0))

        # --- Training hyperparameters ---
        lr         = float(best_cfg["train"]["lr"])
        batch_sz   = int(best_cfg["train"]["batch_size"])
        max_epochs = int(best_cfg["train"].get("epochs", 10))
        val_ratio  = float(best_cfg["train"].get("val_ratio", 0.1))
        es_patience = int(best_cfg["train"].get("early_stop_patience", 10))

        print("[Config] Loaded tuned CNN params from artifacts/best_config.yaml")

    else:
        # --- Default fallback values (reasonable for CNN) ---

        conv1_filters, conv2_filters = map(
            int,
            config['Model']['conv_filters'].split(',')
        )
        dense1_units, dense2_units, dense3_units = map(
            int,
            config['Model']['dense_units'].split(',')
        )

        dropout       = 0.1

        lr         = config['Training'].getfloat('learning_rate', 0.001)
        batch_sz   = config['Training'].getint('batch_size', 128)
        max_epochs = config['Training'].getint('epochs', 50)
        val_ratio  = config['Training'].getfloat('validation_split', 0.1)
        es_patience = config['EarlyStopping'].getint('patience', 10)

        print("[Config] Using default CNN training params")

    # 3) Torch tensors + split
    X_trn, X_val, Y_trn, Y_val = train_test_split(X_tr, Y_tr, test_size=val_ratio, random_state=42, shuffle=True)
    X_trn_t = torch.from_numpy(X_trn);  Y_trn_t = torch.from_numpy(Y_trn)
    X_val_t = torch.from_numpy(X_val);  Y_val_t = torch.from_numpy(Y_val)
    X_te_t  = torch.from_numpy(X_te);   Y_te_t  = torch.from_numpy(Y_te)

    dl_trn = DataLoader(TensorDataset(X_trn_t, Y_trn_t), batch_size=batch_sz, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_sz, shuffle=False)
    dl_te  = DataLoader(TensorDataset(X_te_t,  Y_te_t),  batch_size=batch_sz, shuffle=False)

    # 4) Model/optim
    in_features = X_tr.shape[-1]
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

    # 5) Train with early stopping
    best_val = float("inf"); patience = 0
    ckpt_path = f"{path_results}/artifacts/best.pt"

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

        # Early stopping + model checkpoint
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            patience = 0

            torch.save({
                "model_state": model.state_dict(),
                "in_features": in_features,
                "out_dim": out_dim,
                "conv1_filters": conv1_filters,
                "conv2_filters": conv2_filters,
                "dense1_units": dense1_units,
                "dense2_units": dense2_units,
                "dense3_units": dense3_units,
                "dropout": dropout,
                "optimizer_state": opt.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

        else:
            patience += 1
            if patience >= es_patience:
                print("Early stopping.")
                break


    # 6) Evaluate on test set using the best checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
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
    watershed_name = 'Sugar Creek'
    watershed_name_lower = watershed_name.replace(" ", "_").lower()

    path_ws = Path("/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl")
    cfg_path = f"{path_ws}/tritonlite_{watershed_name_lower}.cfg"

    config = configparser.ConfigParser()
    config.read(cfg_path)
    
    test_set   = "D040"
    train_sets = [f"D{i:03d}" for i in range(1, 41) if f"D{i:03d}" != test_set]  # keep small for CPU tuning

    main()
