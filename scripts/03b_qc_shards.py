"""Quality check of the TRIBE shards before analysis (CPU, a few minutes).

Writes results/analysis/qc/shards.csv (per shard and window) and qc/tasks.csv (per task, answer window):
- completeness and finiteness, value range;
- spread across stimuli (a constant output would mean features were not used);
- duplicated predictions for different stimuli;
- size of the correct-minus-incorrect difference relative to the spread across stimuli.
"""
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.inference import assign_shards
from tribeloc.parcels import load as load_parcels
from tribeloc.rsa import item_differences, item_index


def main():
    cfg = load_config()
    P = cfg["paths"]
    table = pd.read_csv(ROOT / cfg["stimuli"]["output"])
    shards = assign_shards(table, cfg["tribe"]["shard_size"])
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    windows = list(cfg["windows"])
    rows, task_rows, missing = [], [], []
    by_id = table.set_index("stim_id")
    for k in range(shards.max() + 1):
        path = ROOT / P["shards"] / f"shard_{k:04d}.npz"
        if not path.exists():
            missing.append(k)
            continue
        z = np.load(path)
        ids = z["stim_id"]
        expected = set(table.stim_id[shards == k])
        for w in windows:
            x = z[w][:, cortex]
            rows.append(dict(shard=k, window=w, n=len(ids), complete=set(ids) == expected,
                             finite=bool(np.isfinite(x).all()), mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                             sd_across_stimuli=float(x.std(0).mean()),
                             n_duplicate_rows=int(len(x) - len(np.unique(x.round(6), axis=0)))))
        meta = by_id.loc[ids].reset_index()
        x = z["answer"][:, cortex].astype(np.float64)
        for task, g in meta.groupby("task", sort=False):
            idx, _ = item_index(g.reset_index(drop=True))
            xg = x[g.index.to_numpy()]
            d = item_differences(xg, idx)
            task_rows.append(dict(task=task, shard=k, n_items=len(idx),
                                  sd_across_stimuli=float(xg.std(0).mean()), mean_abs_d=float(np.abs(d).mean())))
        print(f"shard {k} ok", flush=True)
    out = ROOT / P["analysis"] / "qc"
    out.mkdir(parents=True, exist_ok=True)
    shards_df = pd.DataFrame(rows)
    shards_df.to_csv(out / "shards.csv", index=False)
    t = pd.DataFrame(task_rows).groupby("task", sort=False).agg(
        n_items=("n_items", "sum"), mean_abs_d=("mean_abs_d", "mean"), sd_across_stimuli=("sd_across_stimuli", "mean")).reset_index()
    t["d_over_sd"] = t.mean_abs_d / t.sd_across_stimuli
    t.to_csv(out / "tasks.csv", index=False)
    print(f"missing shards: {missing}", flush=True)
    print(shards_df.groupby("window")[["finite", "complete"]].all().to_string(), flush=True)
    print(shards_df.groupby("window")[["mean", "sd_across_stimuli", "n_duplicate_rows"]].describe().T.round(4).to_string(), flush=True)
    print(t.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
