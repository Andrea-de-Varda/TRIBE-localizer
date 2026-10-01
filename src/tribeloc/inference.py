"""TRIBE v2 inference: one isolated timeline per stimulus, window means of the predicted response.

Follows the call pattern of the collaborator's pilot (TribeModel.from_pretrained, data.get_loaders,
model._model(batch)). Pure helpers (sharding, events, windows) are separated from the GPU code so
they can be tested without TRIBE installed.
"""
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_VERTICES = 20484
WORD = re.compile(r"\S+")


def assign_shards(table, shard_size):
    """Shard index per row. Shards hold whole items (all four stimuli), in table order."""
    item_key = table.task + "__" + table.item.astype(str)
    item_index = pd.factorize(item_key)[0]
    return item_index // max(1, shard_size // 4)


def build_events(rows, word_seconds, timeline_seconds):
    """TRIBE events for a set of stimulus rows: one timeline per stimulus, one Word event per whitespace word."""
    events = []
    for r in rows.itertuples():
        events.append(dict(type="Event", timeline=r.stim_id, subject="tribeloc", start=0.0, duration=timeline_seconds))
        for j, w in enumerate(WORD.finditer(r.text)):
            events.append(dict(type="Word", timeline=r.stim_id, subject="tribeloc", start=r.onset_s + j * word_seconds,
                               duration=word_seconds, text=w.group(), context=r.text[: w.end()], sentence=r.text,
                               sentence_char=w.start(), language="en", modality="read"))
    return pd.DataFrame(events)


def window_masks(times, row, windows):
    """Boolean mask over output times for each configured window of one stimulus."""
    masks = {}
    for name, w in windows.items():
        if w["anchor"] == "answer_onset":
            lo, hi = row.answer_onset_s + w["start"], row.answer_onset_s + w["end"]
        elif w["anchor"] == "stimulus":
            lo, hi = row.onset_s + w["start"], row.end_s + w["end"]
        else:
            raise ValueError(w["anchor"])
        m = (times >= lo) & (times < hi)
        if not m.any():
            raise ValueError(f"window {name} [{lo}, {hi}) has no samples for {row.stim_id}")
        masks[name] = m
    return masks


def load_model(checkpoint, cache_dir, batch_size, text_batch_size):
    from tribev2 import TribeModel
    cache_dir = Path(cache_dir)
    return TribeModel.from_pretrained(
        checkpoint, cache_folder=str(cache_dir / "features"), cluster=None, device="cuda",
        config_update={"data.features_to_use": ["text"], "data.num_workers": 0, "data.batch_size": batch_size,
                       "data.text_feature.batch_size": text_batch_size, "data.text_feature.infra.cluster": None,
                       "infra.folder": str(cache_dir / "experiment")})


def predict_shard(model, rows, windows, word_seconds, timeline_seconds):
    """Window means (n_rows x N_VERTICES, float32) for each window, in row order."""
    import torch
    events = build_events(rows, word_seconds, timeline_seconds)
    loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
    lookup = {sid: i for i, sid in enumerate(rows.stim_id)}
    rowlist = list(rows.itertuples())
    out = {k: np.zeros((len(rows), N_VERTICES), np.float32) for k in windows}
    done = np.zeros(len(rows), int)
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to("cuda")
            y = model._model(batch).cpu().numpy()  # batch x vertices x time
            for j, seg in enumerate(batch.segments):
                i = lookup[str(seg.timeline)]
                times = seg.start + np.arange(y.shape[-1]) * model.data.TR
                for name, m in window_masks(times, rowlist[i], windows).items():
                    out[name][i] = y[j][:, m].mean(1)
                done[i] += 1
    if not (done == 1).all():
        raise RuntimeError(f"{(done != 1).sum()} stimuli not predicted exactly once")
    for name, x in out.items():
        if not np.isfinite(x).all():
            raise RuntimeError(f"non-finite predictions in window {name}")
    return out


def run_shards(table, shard_ids, cfg, out_dir, cache_root):
    """Predict the given shards, skipping those already saved. Returns per-shard timing records."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s, t = cfg["stimuli"], cfg["tribe"]
    shards = assign_shards(table, t["shard_size"])
    log = []
    for k in shard_ids:
        path = out_dir / f"shard_{k:04d}.npz"
        if path.exists():
            continue
        rows = table[shards == k].reset_index(drop=True)
        if rows.empty:
            continue
        # Fresh model and node-local feature cache per shard; the cache is deleted afterwards
        # (TRIBE caches 20 Llama layers per word, about 245 KB).
        cache = Path(cache_root) / f"shard_{k:04d}"
        start = time.time()
        model = load_model(t["checkpoint"], cache, t["batch_size"], t["text_batch_size"])
        maps = predict_shard(model, rows, cfg["windows"], s["word_seconds"], s["timeline_seconds"])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, stim_id=rows.stim_id.to_numpy().astype(str), **maps)
        tmp.rename(path)
        del model
        shutil.rmtree(cache, ignore_errors=True)
        rec = dict(shard=k, n_stimuli=len(rows), n_words=int(rows.n_words.sum()), seconds=time.time() - start)
        log.append(rec)
        print(f"SHARD {k}: {rec}", flush=True)
    return log
