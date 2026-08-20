#!/usr/bin/env python3
"""Compare held-out Stage-1 metrics from evaluation payloads."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage1-evaluation")
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


def payload(path):
    data=json.loads(path.read_text())
    return data["test"]


def main():
    args=parse_args(); old=payload(args.previous); new=payload(args.current); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    panels=[
        (["depth_all_rmse","depth_wet_rmse"],["All depth","Wet depth"],"Depth RMSE","m"),
        (["component_mae","component_rmse"],["MAE","RMSE"],"Velocity-component error","m/s"),
        (["wet_precision","wet_recall","wet_f1","wet_csi"],["Precision","Recall","F1","CSI"],"Inundation skill","Score"),
    ]
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True); rows=[]
    for ax,(metrics,labels,title,ylabel) in zip(axes,panels):
        x=np.arange(len(metrics)); w=.36
        a=ax.bar(x-w/2,[old[k] for k in metrics],w,label="Previous stratified",color="#777777")
        b=ax.bar(x+w/2,[new[k] for k in metrics],w,label="Current",color="#2878B5")
        ax.bar_label(a,fmt="%.3f",padding=3,fontsize=8); ax.bar_label(b,fmt="%.3f",padding=3,fontsize=8)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,title=title); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.3)
        for k in metrics:
            higher=k.startswith("wet_")
            improvement=100*((new[k]-old[k]) if higher else (old[k]-new[k]))/old[k]
            rows.append({"metric":k,"previous":old[k],"current":new[k],"relative_improvement_percent":improvement})
    axes[0].legend(frameon=False); fig.suptitle("Held-out D030: identical 1,000-batch evaluation",fontsize=15,fontweight="bold")
    fig.savefig(output/"04_D030_fair_test_comparison.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    with (output/"D030_fair_test_comparison.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote held-out comparison to {output}")


if __name__ == "__main__":
    main()
