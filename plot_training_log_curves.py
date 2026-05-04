import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EPOCH_PATTERN = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train_loss>[-+0-9.eE]+)\s+"
    r"train_rmse=(?P<train_rmse>[-+0-9.eE]+)\s+"
    r"val_loss=(?P<val_loss>[-+0-9.eE]+)\s+"
    r"val_rmse=(?P<val_rmse>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss/RMSE curves from Slurm log output")
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_history(train_log: Path) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for line in train_log.read_text().splitlines():
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        records.append(
            {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train_loss")),
                "train_rmse": float(match.group("train_rmse")),
                "val_loss": float(match.group("val_loss")),
                "val_rmse": float(match.group("val_rmse")),
            }
        )

    if not records:
        raise ValueError(f"No epoch metric lines were found in {train_log}")

    return pd.DataFrame(records).sort_values("epoch").reset_index(drop=True)


def plot_curves(history: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="val", linewidth=2)
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked Huber Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_rmse"], label="train", linewidth=2)
    axes[1].plot(history["epoch"], history["val_rmse"], label="val", linewidth=2)
    axes[1].set_title("RMSE by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Masked RMSE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    out_path = output_dir / "training_curves_from_log.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = parse_history(args.train_log.resolve())
    history.to_csv(output_dir / "training_history_from_log.csv", index=False)
    plot_curves(history, output_dir)

    print(f"Wrote training curves to {output_dir}")


if __name__ == "__main__":
    main()
