#!/usr/bin/env python3
"""Plot truth, prediction, and error for selected Stage-1 samples."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stage1-predictions")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wet-threshold", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def masked(array, mask):
    return np.where(mask, array, np.nan)


def main():
    args = parse_args(); source=args.prediction_dir.resolve(); output=args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    records=json.loads((source/"prediction_manifest.json").read_text())
    summaries=[]
    for record in records:
        data=np.load(record["path"]); mask=data["mask"]>0.5
        true_d=data["true_depth"]; pred_d=data["depth"]
        true_s=np.hypot(data["true_component_x"],data["true_component_y"]); pred_s=np.hypot(data["component_x"],data["component_y"])
        wet=true_d>=args.wet_threshold; pred_wet=data["wet_probability"]>=0.5
        vmax=max(float(np.nanpercentile(masked(true_d,mask),99)),float(np.nanpercentile(masked(pred_d,mask),99)),args.wet_threshold)
        smax=max(float(np.nanpercentile(masked(true_s,mask),99)),float(np.nanpercentile(masked(pred_s,mask),99)),1e-3)
        fig,axes=plt.subplots(3,3,figsize=(12,11),constrained_layout=True)
        panels=[
            (true_d,"True depth","viridis",0,vmax),(pred_d,"Predicted depth","viridis",0,vmax),(pred_d-true_d,"Depth error","coolwarm",-vmax,vmax),
            (true_s,"True speed","magma",0,smax),(pred_s,"Predicted speed","magma",0,smax),(pred_s-true_s,"Speed error","coolwarm",-smax,smax),
            (wet.astype(float),"True wet mask","Blues",0,1),(data["wet_probability"],"Predicted wet probability","Blues",0,1),(pred_wet.astype(float)-wet.astype(float),"Wet-mask error","coolwarm",-1,1),
        ]
        for ax,(arr,title,cmap,vmin,vmax_panel) in zip(axes.flat,panels):
            image=ax.imshow(masked(arr,mask),cmap=cmap,vmin=vmin,vmax=vmax_panel); ax.set_title(title); ax.axis("off"); fig.colorbar(image,ax=ax,shrink=.72)
        stem=Path(record["path"]).stem
        fig.suptitle(stem,fontsize=15,fontweight="bold"); fig.savefig(output/f"{stem}.png",dpi=args.dpi,bbox_inches="tight"); plt.close(fig)
        valid_depth=(pred_d-true_d)[mask]; true_wet=wet&mask
        summaries.append({"event_id":record["event_id"],"time_index":record["time_index"],"block_position":record["block_position"],"depth_mae":float(np.abs(valid_depth).mean()),"depth_rmse":float(np.sqrt(np.mean(valid_depth**2))),"wet_depth_rmse":float(np.sqrt(np.mean((pred_d[true_wet]-true_d[true_wet])**2))) if true_wet.any() else None,"wet_fraction":float(true_wet.sum()/max(mask.sum(),1))})
    (output/"sample_metrics.json").write_text(json.dumps(summaries,indent=2)); print(f"Wrote {len(records)} sample figures to {output}")


if __name__ == "__main__":
    main()
