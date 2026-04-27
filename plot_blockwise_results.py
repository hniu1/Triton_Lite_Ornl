import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot blockwise training results")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_metrics(results_dir: Path) -> dict:
    return json.loads((results_dir / "metrics.json").read_text())


def plot_training_curves(metrics: dict, output_dir: Path) -> None:
    history = metrics["history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    train_rmse = [row["train"]["rmse"] for row in history]
    val_rmse = [row["val"]["rmse"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="train", linewidth=2)
    axes[0].plot(epochs, val_loss, label="val", linewidth=2)
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_rmse, label="train", linewidth=2)
    axes[1].plot(epochs, val_rmse, label="val", linewidth=2)
    axes[1].set_title("RMSE by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_split_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    split_dir = results_dir / "splits"
    return {
        "train": pd.read_csv(split_dir / "train_samples.csv"),
        "val": pd.read_csv(split_dir / "val_samples.csv"),
        "test": pd.read_csv(split_dir / "test_samples.csv"),
    }


def plot_split_sizes(split_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    labels = list(split_tables.keys())
    sample_counts = [len(split_tables[label]) for label in labels]
    event_counts = [split_tables[label]["event_id"].nunique() for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(labels, sample_counts, color=["#355070", "#6d597a", "#b56576"])
    axes[0].set_title("Samples per Split")
    axes[0].set_ylabel("Rows")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, event_counts, color=["#355070", "#6d597a", "#b56576"])
    axes[1].set_title("Events per Split")
    axes[1].set_ylabel("Unique Events")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "split_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_distributions(split_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, frame in split_tables.items():
        axes[0].hist(frame["y"], bins=50, alpha=0.45, label=label, density=True)
        positive = frame.loc[frame["y"] > 0, "y"]
        if not positive.empty:
            axes[1].hist(positive, bins=50, alpha=0.45, label=label, density=True)

    axes[0].set_title("Target Distribution")
    axes[0].set_xlabel("Flood Depth y")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Positive-Depth Distribution")
    axes[1].set_xlabel("Flood Depth y | y > 0")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "target_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_zero_fraction(split_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    labels = list(split_tables.keys())
    zero_fraction = [(frame["y"] == 0).mean() for frame in split_tables.values()]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, zero_fraction, color=["#355070", "#6d597a", "#b56576"])
    ax.set_ylim(0, 1)
    ax.set_title("Zero-Depth Share by Split")
    ax.set_ylabel("Fraction of rows with y = 0")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "zero_fraction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "plots")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(results_dir)
    split_tables = load_split_tables(results_dir)

    plot_training_curves(metrics, output_dir)
    plot_split_sizes(split_tables, output_dir)
    plot_target_distributions(split_tables, output_dir)
    plot_zero_fraction(split_tables, output_dir)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()