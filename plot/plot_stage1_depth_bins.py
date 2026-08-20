#!/usr/bin/env python3
"""Plot previous-versus-current Stage-1 metrics by true-depth bin."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage1-depth-bins")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--current",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    return parser.parse_args()


def main():
    args=parse_args(); old=json.loads(args.previous.read_text())["depth_bins"]; new=json.loads(args.current.read_text())["depth_bins"]
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True); names=list(new); labels=["Dry\n<0.05","Shallow\n0.05–0.25","Moderate\n0.25–1","Deep\n1–2","Extreme\n≥2 m"]
    panels=[("depth_rmse","Depth RMSE","m"),("depth_bias","Depth bias","m"),("component_rmse","Velocity-component RMSE","m/s"),("speed_rmse","Speed RMSE","m/s"),("direction_mae_degrees","Direction MAE (true speed ≥0.05 m/s)","degrees"),("predicted_wet_rate","Predicted-wet rate","fraction")]
    x=np.arange(len(names)); width=.36; fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    rows=[]
    for ax,(metric,title,ylabel) in zip(axes.flat,panels):
        old_values=[old[name][metric] if metric != "direction_mae_degrees" or old[name]["direction_cell_count"] >= 100 else np.nan for name in names]
        new_values=[new[name][metric] if metric != "direction_mae_degrees" or new[name]["direction_cell_count"] >= 100 else np.nan for name in names]
        ax.bar(x-width/2,old_values,width,label="Previous stratified",color="#777777"); ax.bar(x+width/2,new_values,width,label="Current",color="#2878B5")
        if metric=="depth_bias": ax.axhline(0,color="black",linewidth=.8)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,title=title); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.3)
        for name,old_value,new_value in zip(names,old_values,new_values): rows.append({"depth_bin":name,"metric":metric,"previous":old_value,"current":new_value,"change":new_value-old_value})
    axes.flat[0].legend(frameon=False); fig.suptitle("Held-out D030 errors by true-depth bin",fontsize=15,fontweight="bold")
    fig.savefig(output/"05_D030_metrics_by_depth_bin.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    with (output/"D030_metrics_by_depth_bin.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    distribution=[new[name]["valid_cell_fraction"]*100 for name in names]
    fig,ax=plt.subplots(figsize=(9,5.2),constrained_layout=True); bars=ax.bar(x,distribution,color="#2878B5"); ax.bar_label(bars,fmt="%.2f%%",padding=3); ax.set(xticks=x,xticklabels=labels,ylabel="Valid D030 cells (%)",title="D030 evaluation depth distribution"); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.3); fig.savefig(output/"06_D030_depth_bin_distribution.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"Wrote depth-bin diagnostics to {output}")


if __name__ == "__main__":
    main()
