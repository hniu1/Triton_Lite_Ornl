"""Native-cell depth and rising-water diagnostics, separate from extent scores."""
import math
import torch


class DepthDiagnostics:
    def __init__(self):
        self.sums = {}

    def update(self, prediction, target, origin, valid):
        valid = (valid > .5) & torch.isfinite(prediction) & torch.isfinite(target) & torch.isfinite(origin)
        error = prediction - target
        groups = {f'{lo:g}_to_{hi:g}m': (target >= lo) & (target < hi)
                  for lo,hi in [(.05,.25),(.25,1),(1,2),(2,5),(5,float('inf'))]}
        groups['deep_rising_ge1m_change_ge0.1m'] = (target >= 1) & (target-origin >= .1)
        for name, selected in groups.items():
            e=error[valid & selected].double()
            values=self.sums.setdefault(name,[0,0.,0.,0.,0,0])
            values[0]+=e.numel()
            values[1]+=float(e.sum()); values[2]+=float(e.abs().sum()); values[3]+=float(e.square().sum())
            values[4]+=int((e < -.3).sum()); values[5]+=int((e > .3).sum())

    def finalize(self):
        return {name:dict(cells=n,bias_m=s/n if n else None,mae_m=a/n if n else None,
                         rmse_m=math.sqrt(q/n) if n else None,
                         underprediction_gt0_3m_fraction=u/n if n else None,
                         overprediction_gt0_3m_fraction=o/n if n else None)
                for name,(n,s,a,q,u,o) in self.sums.items()}
