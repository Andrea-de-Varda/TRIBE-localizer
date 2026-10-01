"""Univariate maps: per-task mean response over all stimuli, relative to the window-matched no-input baseline
(12_baseline.py), and each domain minus the other three (tasks weighted equally) with task-label permutation
FWE. Writes results/analysis/univariate.npz (task_means_<w>: baseline-subtracted; task_means_raw_<w>: raw)."""
from pathlib import Path

import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.baseline import group_baselines
from tribeloc.group import DOMAINS, univariate_permutation
from tribeloc.inference import N_VERTICES, assign_shards
from tribeloc.parcels import load as load_parcels


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
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    means = task_means(table, uc["windows"], ROOT / P["shards"], cfg["tribe"]["shard_size"])
    out = dict(domains=np.array(DOMAINS, dtype=str), task_domains=domains.astype(str))
    bl = cfg["baseline"]
    if bl["subtract"]:
        z = np.load(ROOT / bl["path"])                              # fails if 12_baseline.py has not been run
        wins = {w: cfg["windows"][w] for w in uc["windows"]}
        base = group_baselines(table, table.task_index.to_numpy(), len(domains), z["timecourse"], z["times"], wins)
    rng = np.random.default_rng(uc["seed"])
    for w, raw in means.items():
        m = raw - base[w] if bl["subtract"] else raw
        contrast, p = univariate_permutation(m, domains, cortex, uc["n_permutations"], rng)
        out[f"task_means_{w}"], out[f"contrast_{w}"], out[f"p_fwe_{w}"] = m.astype(np.float32), contrast, p
        out[f"task_means_raw_{w}"] = raw.astype(np.float32)
        for i, d in enumerate(DOMAINS):
            print(w, d, f"FWE p<.05 vertices: {(p[i] < .05).sum()}", flush=True)
    np.savez_compressed(ROOT / P["analysis"] / "univariate.npz", **out)


if __name__ == "__main__":
    main()
