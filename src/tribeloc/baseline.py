"""No-input baseline: TRIBE's prediction when its text features are all zero.

TRIBE was trained on BOLD z-scored per vertex and run during continuous naturalistic stimulation, so its output
has no rest or fixation reference. Zero text features are what the model receives at every time point without a
word, so the zero-input prediction is the computational counterpart of a no-stimulus baseline. It is not human
rest: the training data contain no rest periods.

With zero features every timeline is identical (same subject, same 100-s segment), so the baseline depends only
on time within the segment. One prediction gives the full time course; each stimulus's baseline is that time
course averaged over the stimulus's own analysis window.
"""
import numpy as np

from tribeloc.inference import build_events, window_masks


def predict_no_input(model, rows, word_seconds, timeline_seconds):
    """Zero-feature prediction (vertices x time) and output times, for one or more dummy stimulus rows.
    Returns a list of (prediction, times), one per row, so the caller can check they are identical."""
    import torch
    events = build_events(rows, word_seconds, timeline_seconds)
    loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
    out = {}
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to("cuda")
            batch.data["text"] = torch.zeros_like(batch.data["text"])
            y = model._model(batch).cpu().numpy()
            for j, seg in enumerate(batch.segments):
                out[str(seg.timeline)] = (y[j], seg.start + np.arange(y.shape[-1]) * model.data.TR)
    return [out[s] for s in rows.stim_id]


def window_keys(rows, times, windows):
    """For every row and window, an index into the distinct output-sample masks (rows sharing onset and end
    share a mask). Returns {window: (row_key, masks)} with masks a (n_keys x n_times) boolean array."""
    out = {}
    for w in windows:
        keys, masks, row_key = {}, [], np.empty(len(rows), int)
        for i, r in enumerate(rows.itertuples()):
            m = window_masks(times, r, {w: windows[w]})[w]
            k = m.tobytes()
            if k not in keys:
                keys[k] = len(masks)
                masks.append(m)
            row_key[i] = keys[k]
        out[w] = (row_key, np.array(masks))
    return out


def stimulus_baselines(rows, tc, times, windows):
    """Per-row baseline (n_rows x V) for each window: the no-input time course tc (V x T) averaged over the row's
    window. Use for small tables (localizers); for the task table use group_baselines."""
    out = {}
    for w, (row_key, masks) in window_keys(rows, times, windows).items():
        maps = np.stack([tc[:, m].mean(1) for m in masks])          # n_keys x V
        out[w] = maps[row_key]
    return out


def group_baselines(rows, groups, n_groups, tc, times, windows):
    """Mean baseline per group (n_groups x V) for each window, without materializing one row per stimulus."""
    out = {}
    for w, (row_key, masks) in window_keys(rows, times, windows).items():
        maps = np.stack([tc[:, m].mean(1) for m in masks])          # n_keys x V
        counts = np.zeros((n_groups, len(masks)))
        np.add.at(counts, (np.asarray(groups), row_key), 1)
        out[w] = (counts / counts.sum(1, keepdims=True)) @ maps
    return out
