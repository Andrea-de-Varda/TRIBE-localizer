"""Univariate maps: per-task mean response over all stimuli (and over each item half, for the held-out
parcel fROIs), and each domain minus the other three (tasks weighted equally) with task-label
permutation FWE. Writes results/analysis/univariate.npz."""
from pathlib import Path

import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.group import DOMAINS, univariate_permutation
from tribeloc.inference import N_VERTICES, assign_shards
from tribeloc.parcels import load as load_parcels


def task_means(table, windows, shard_dir, shard_size):
    """Mean prediction per task and item half, reading every shard once.
    Returns {window: (all: 46 x V, by_half: 46 x 2 x V)}; halves are 1 and 2 in the table."""
    shards = assign_shards(table, shard_size)
    task_of = dict(zip(table.stim_id, table.task_index))
    half_of = dict(zip(table.stim_id, table.half - 1))
    n_tasks = table.task_index.nunique()
    sums = {w: np.zeros((n_tasks, 2, N_VERTICES)) for w in windows}
    counts = np.zeros((n_tasks, 2))
    for k in np.unique(shards):
        z = np.load(Path(shard_dir) / f"shard_{k:04d}.npz")
        t = np.array([task_of[s] for s in z["stim_id"]])
        h = np.array([half_of[s] for s in z["stim_id"]])
        for w in windows:
            np.add.at(sums[w], (t, h), z[w].astype(np.float64))
        np.add.at(counts, (t, h), 1)
    assert counts.sum() == len(table), "shards incomplete"
    return {w: (s.sum(1) / counts.sum(1)[:, None], s / counts[:, :, None]) for w, s in sums.items()}


def main():
    cfg = load_config()
    uc, P = cfg["univariate"], cfg["paths"]
    table = pd.read_csv(ROOT / cfg["stimuli"]["output"])
    domains = table.groupby("task_index").domain.first().sort_index().to_numpy()
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    means = task_means(table, uc["windows"], ROOT / P["shards"], cfg["tribe"]["shard_size"])
    out = dict(domains=np.array(DOMAINS, dtype=str), task_domains=domains.astype(str))
    rng = np.random.default_rng(uc["seed"])
    for w, (m, by_half) in means.items():
        contrast, p = univariate_permutation(m, domains, cortex, uc["n_permutations"], rng)
        out[f"task_means_{w}"], out[f"contrast_{w}"], out[f"p_fwe_{w}"] = m.astype(np.float32), contrast, p
        out[f"task_means_{w}_h1"], out[f"task_means_{w}_h2"] = by_half[:, 0].astype(np.float32), by_half[:, 1].astype(np.float32)
        for i, d in enumerate(DOMAINS):
            print(w, d, f"FWE p<.05 vertices: {(p[i] < .05).sum()}", flush=True)
    np.savez_compressed(ROOT / P["analysis"] / "univariate.npz", **out)


if __name__ == "__main__":
    main()
