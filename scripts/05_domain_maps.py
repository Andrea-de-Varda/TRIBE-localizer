"""Domain crossnobis and accuracy maps (tasks weighted equally) with max-statistic FWE p-values.

The domain null averages the per-task sign-flip null maps permutation by permutation (flips are
independent across tasks). Writes results/analysis/domain_maps.npz.
"""
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.group import DOMAINS, fwe_p
from tribeloc.parcels import load as load_parcels


def main():
    cfg = load_config()
    P = cfg["paths"]
    tasks = pd.read_csv(ROOT / "data" / "stimuli" / "task_summary.csv")
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    out = {}
    for d in DOMAINS:
        names = tasks[tasks.domain == d].task.tolist()
        res = [np.load(ROOT / P["analysis"] / "searchlight" / f"{t}.npz") for t in names]
        null = np.mean([np.load(ROOT / P["permutations"] / f"{t}.npy") for t in names], axis=0)
        u = np.mean([r["crossnobis"] for r in res], axis=0)
        p = np.full(len(cortex), np.nan)
        p[cortex] = fwe_p(u[cortex], null)
        out[f"{d}_crossnobis"], out[f"{d}_p_fwe"] = u, p
        if "accuracy" in res[0].files:
            out[f"{d}_accuracy"] = np.mean([r["accuracy"] for r in res], axis=0)
        print(d, f"{len(names)} tasks, max crossnobis {np.nanmax(u):.4g}, FWE p<.05 vertices: {(p < .05).sum()}", flush=True)
    np.savez_compressed(ROOT / P["analysis"] / "domain_maps.npz", **out)


if __name__ == "__main__":
    main()
