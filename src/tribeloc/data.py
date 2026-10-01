"""Load TRIBE window means for a subset of the stimulus table from the saved shards."""
from pathlib import Path

import numpy as np

from tribeloc.inference import N_VERTICES, assign_shards


def load_predictions(table, rows, window, shard_dir, shard_size):
    """(len(rows) x N_VERTICES) float32 predictions for `rows` (a subset of `table`, matched by stim_id)."""
    shards = assign_shards(table, shard_size)[table.index.get_indexer(rows.index)]
    want = {sid: i for i, sid in enumerate(rows.stim_id)}
    out = np.full((len(rows), N_VERTICES), np.nan, np.float32)
    for k in np.unique(shards):
        z = np.load(Path(shard_dir) / f"shard_{k:04d}.npz")
        ids, arr = z["stim_id"], z[window]
        for j, sid in enumerate(ids):
            i = want.get(sid)
            if i is not None:
                out[i] = arr[j]
    if np.isnan(out).any():
        raise RuntimeError(f"{int(np.isnan(out).any(1).sum())} stimuli missing from shards")
    return out
