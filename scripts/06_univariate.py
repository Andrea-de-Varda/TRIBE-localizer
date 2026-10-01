"""Task means (needs the shards): per-task mean response over all stimuli, for each window and response reference.
  raw_<w>      raw TRIBE prediction (0 = the vertex's mean during naturalistic stimulation in training)
  noinput_<w>  minus the window-matched no-input baseline (12_baseline.py)
Writes results/analysis/task_means.npz. Contrast maps are computed from it by 07_univariate_contrasts.py."""
from pathlib import Path

import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.baseline import group_baselines
from tribeloc.inference import N_VERTICES, assign_shards


def task_means(table, windows, shard_dir, shard_size):
    """Mean prediction per task (46 x V) for each window, reading every shard once."""
    shards = assign_shards(table, shard_size)
    task_of = dict(zip(table.stim_id, table.task_index))
    n_tasks = table.task_index.nunique()
    sums = {w: np.zeros((n_tasks, N_VERTICES)) for w in windows}
    counts = np.zeros(n_tasks)
    for k in np.unique(shards):
        z = np.load(Path(shard_dir) / f"shard_{k:04d}.npz")
        t = np.array([task_of[s] for s in z["stim_id"]])
        for w in windows:
            np.add.at(sums[w], t, z[w].astype(np.float64))
        np.add.at(counts, t, 1)
    assert counts.sum() == len(table), "shards incomplete"
    return {w: s / counts[:, None] for w, s in sums.items()}


def main():
    cfg = load_config()
    uc, P = cfg["univariate"], cfg["paths"]
    table = pd.read_csv(ROOT / cfg["stimuli"]["output"])
    domains = table.groupby("task_index").domain.first().sort_index().to_numpy()
    means = task_means(table, uc["windows"], ROOT / P["shards"], cfg["tribe"]["shard_size"])
    z = np.load(ROOT / cfg["baseline"]["path"])                     # fails if 12_baseline.py has not been run
    base = group_baselines(table, table.task_index.to_numpy(), len(domains), z["timecourse"], z["times"],
                           {w: cfg["windows"][w] for w in uc["windows"]})
    out = dict(task_domains=domains.astype(str))
    for w, raw in means.items():
        out[f"raw_{w}"], out[f"noinput_{w}"] = raw.astype(np.float32), (raw - base[w]).astype(np.float32)
    np.savez_compressed(ROOT / P["analysis"] / "task_means.npz", **out)
    print("task means saved:", sorted(out), flush=True)


if __name__ == "__main__":
    main()
