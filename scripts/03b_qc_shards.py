"""Quality check of the TRIBE shards before analysis (CPU, a few minutes).

Writes results/analysis/qc/shards.csv (per shard and window):
- completeness and finiteness, value range;
- spread across stimuli (a constant output would mean features were not used);
- duplicated predictions for different stimuli.
"""
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.inference import assign_shards
from tribeloc.parcels import load as load_parcels


def main():
    cfg = load_config()
    P = cfg["paths"]
    table = pd.read_csv(ROOT / cfg["stimuli"]["output"])
    shards = assign_shards(table, cfg["tribe"]["shard_size"])
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    windows = list(cfg["windows"])
    rows, missing = [], []
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
        print(f"shard {k} ok", flush=True)
    out = ROOT / P["analysis"] / "qc"
    out.mkdir(parents=True, exist_ok=True)
    shards_df = pd.DataFrame(rows)
    shards_df.to_csv(out / "shards.csv", index=False)
    print(f"missing shards: {missing}", flush=True)
    print(shards_df.groupby("window")[["finite", "complete"]].all().to_string(), flush=True)
    print(shards_df.groupby("window")[["mean", "sd_across_stimuli", "n_duplicate_rows"]].describe().T.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
