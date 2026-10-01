"""Localizer contrasts and fROI definition (top fraction of each parcel's vertices by the localizer t)."""
import numpy as np

from tribeloc.group import top_vertices


def contrast_t(x_target, x_control, test, pairs_target=None, pairs_control=None):
    """Per-vertex t for target > control. welch: independent samples; paired: rows matched by pair id."""
    if test == "welch":
        a, b = x_target, x_control
        se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        return (a.mean(0) - b.mean(0)) / np.maximum(se, 1e-12)
    if test == "paired":
        ia, ib = np.argsort(pairs_target), np.argsort(pairs_control)
        assert np.array_equal(np.asarray(pairs_target)[ia], np.asarray(pairs_control)[ib]), "unmatched pairs"
        d = x_target[ia] - x_control[ib]
        return d.mean(0) / np.maximum(d.std(0, ddof=1) / np.sqrt(len(d)), 1e-12)
    raise ValueError(test)


def localizer_t(X, rows, spec, mask=None):
    """t map for one localizer contrast; rows: localizer table rows aligned with X; mask: optional row subset."""
    m = np.ones(len(rows), bool) if mask is None else np.asarray(mask)
    t_rows = m & (rows.condition == spec["target"]).to_numpy()
    c_rows = m & (rows.condition == spec["control"]).to_numpy()
    return contrast_t(X[t_rows], X[c_rows], spec["test"], rows.pair.to_numpy()[t_rows], rows.pair.to_numpy()[c_rows])


def define_frois(t_map, parcel_labels, cortex, labels, fraction):
    """{label: vertex indices}: the top `fraction` of each parcel's cortical vertices by t_map."""
    return {lab: top_vertices(t_map, np.flatnonzero((parcel_labels == lab) & cortex), fraction) for lab in labels}


def heldout_effect(X, rows, spec, parcel_vertices, fraction):
    """Split-half localizer validation for one parcel: select the top vertices on one half of the localizer
    stimuli, measure the target-minus-control difference of the other half's stimuli there; both directions.
    Returns the mean difference over the two directions and the mean held-out t over stimuli."""
    effects, ts = [], []
    halves = rows.half.to_numpy()
    for sel, ev in [(1, 2), (2, 1)]:
        t_sel = localizer_t(X[:, parcel_vertices], rows, spec, halves == sel)
        chosen = parcel_vertices[np.argsort(t_sel, kind="stable")[-max(1, int(np.ceil(fraction * len(parcel_vertices)))):]]
        v = X[:, chosen].mean(1, keepdims=True)
        evm = halves == ev
        tgt = evm & (rows.condition == spec["target"]).to_numpy()
        ctl = evm & (rows.condition == spec["control"]).to_numpy()
        effects.append(v[tgt, 0].mean() - v[ctl, 0].mean())
        ts.append(float(contrast_t(v[tgt], v[ctl], spec["test"], rows.pair.to_numpy()[tgt], rows.pair.to_numpy()[ctl])[0]))
    return float(np.mean(effects)), float(np.mean(ts))
