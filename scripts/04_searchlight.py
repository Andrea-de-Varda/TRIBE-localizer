"""Searchlight crossnobis (all items, each half, sign-flip null) and matched-N classification for one task.

    python scripts/04_searchlight.py --task-index 0      # 0..45, as in the Slurm array

Writes results/analysis/searchlight/<task>.npz (tracked) and results/tribe/perm/<task>.npy (cluster only).
"""
import argparse
import time

import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.data import load_predictions
from tribeloc.parcels import load as load_parcels
from tribeloc.rsa import item_differences, item_index, searchlight_accuracy, searchlight_crossnobis
from tribeloc.surface import searchlights


def to_full(values, cortex):
    out = np.full(len(cortex), np.nan, np.float32)
    out[cortex] = values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--skip-classification", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    P, sc, cc = cfg["paths"], cfg["searchlight"], cfg["classification"]
    table = pd.read_csv(ROOT / cfg["stimuli"]["output"])
    rows = table[table.task_index == a.task_index]
    task = rows.task.iloc[0]
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    start = time.time()
    print(f"{task}: {rows.item.nunique()} items, {len(rows)} stimuli", flush=True)
    sl = searchlights(cortex, sc["radius_mm"])

    X = load_predictions(table, rows, sc["window"], ROOT / P["shards"], cfg["tribe"]["shard_size"])
    idx, items = item_index(rows.reset_index(drop=True))
    half = rows.groupby("item").half.first().loc[items].to_numpy()
    D = item_differences(X, idx).astype(np.float64)
    print(f"{task}: predictions loaded ({time.time() - start:.0f} s)", flush=True)

    rng = np.random.default_rng([sc["seed"], a.task_index])
    signs = rng.choice([-1.0, 1.0], size=(sc["n_permutations"], len(items)))
    u_all, null = searchlight_crossnobis(D, sl, signs)
    u_h1, _ = searchlight_crossnobis(D[half == 1], sl)
    u_h2, _ = searchlight_crossnobis(D[half == 2], sl)
    maxima = np.sort(null.max(1))
    p_fwe = (1 + len(maxima) - np.searchsorted(maxima, u_all, side="left")) / (1 + len(maxima))
    p_unc = (1 + (null >= u_all[None]).sum(0)) / (1 + len(null))
    print(f"{task}: crossnobis done in {time.time() - start:.0f} s", flush=True)

    out = dict(task=task, n_items=len(items), n_items_h1=int((half == 1).sum()), n_items_h2=int((half == 2).sum()),
               crossnobis=to_full(u_all, cortex), crossnobis_h1=to_full(u_h1, cortex), crossnobis_h2=to_full(u_h2, cortex),
               p_fwe=to_full(p_fwe, cortex), p_unc=to_full(p_unc, cortex), null_max=maxima.astype(np.float32))
    if not a.skip_classification:
        crng = np.random.default_rng([cc["seed"], a.task_index])
        Xc = X.astype(np.float64)
        acc, acc_sd = searchlight_accuracy(Xc, idx, sl, cc["n_items"], cc["n_repeats"], cc["n_folds"], crng)
        out.update(accuracy=to_full(acc, cortex), accuracy_sd=to_full(acc_sd, cortex))
        print(f"{task}: classification done in {time.time() - start:.0f} s", flush=True)

    for d in (ROOT / P["analysis"] / "searchlight", ROOT / P["permutations"]):
        d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / P["analysis"] / "searchlight" / f"{task}.npz", **out)
    np.save(ROOT / P["permutations"] / f"{task}.npy", null)
    print(f"{task}: saved ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
