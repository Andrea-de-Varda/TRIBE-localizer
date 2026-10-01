"""Group-level statistics on vertex maps: domain combination, univariate contrasts,
held-out parcel fROIs and spin tests."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DOMAINS = ["Lan", "MD", "phys", "ToM"]


def fwe_p(observed, null):
    """Max-statistic FWE p per vertex: (1 + #{perm max >= observed}) / (1 + n_perm). One-sided."""
    maxima = np.sort(np.nanmax(null, axis=1))
    count = len(maxima) - np.searchsorted(maxima, observed, side="left")
    return (1 + count) / (1 + len(maxima))


def domain_contrast(task_maps, domains):
    """Each domain's mean task map minus the mean of the other three domain means (tasks weighted equally)."""
    means = np.stack([task_maps[domains == d].mean(0) for d in DOMAINS])
    return np.stack([means[i] - np.delete(means, i, 0).mean(0) for i in range(len(DOMAINS))])


def univariate_permutation(task_maps, domains, cortex, n_perm, rng):
    """Observed domain contrasts (4 x V, NaN outside cortex) and one-sided max-statistic FWE p-values
    (maximum over cortical vertices) from shuffling the task-to-domain labels."""
    x = task_maps[:, cortex]
    obs = domain_contrast(x, domains)
    null = np.empty((len(DOMAINS), n_perm))
    for p in range(n_perm):
        null[:, p] = domain_contrast(x, rng.permutation(domains)).max(1)
    contrast = np.full((len(DOMAINS), len(cortex)), np.nan)
    p_fwe = np.full((len(DOMAINS), len(cortex)), np.nan)
    contrast[:, cortex] = obs
    p_fwe[:, cortex] = np.stack([fwe_p(obs[i], null[i][:, None]) for i in range(len(DOMAINS))])
    return contrast, p_fwe


def top_vertices(values, vertices, fraction):
    """The ceil(fraction * n) vertices with the largest values (stable order)."""
    k = max(1, int(np.ceil(fraction * len(vertices))))
    return vertices[np.argsort(values[vertices], kind="stable")[-k:]]


def heldout_froi(map_h1, map_h2, vertices, fraction):
    """Select the top vertices on one half, evaluate the mean on the other, both directions, averaged."""
    a = map_h2[top_vertices(map_h1, vertices, fraction)].mean()
    b = map_h1[top_vertices(map_h2, vertices, fraction)].mean()
    return (a + b) / 2


def parcel_table(results, tasks, parcels, audit, cortex, fraction):
    """Held-out fROI and whole-parcel crossnobis for every task x parcel.

    results: dict task -> npz-like with 'crossnobis_h1', 'crossnobis_h2', 'crossnobis' (full vertex maps)
    tasks: DataFrame with task, domain; parcels: network -> labels; audit: parcel table.
    """
    rows = []
    for p in audit.itertuples():
        vertices = np.flatnonzero((parcels[p.network] == p.label) & cortex)
        for t in tasks.itertuples():
            r = results[t.task]
            rows.append(dict(network=p.network, parcel=p.name, hemisphere=p.hemisphere, n_vertices=len(vertices),
                             task=t.task, domain=t.domain,
                             froi_heldout=heldout_froi(r["crossnobis_h1"], r["crossnobis_h2"], vertices, fraction),
                             whole_parcel=float(np.nanmean(r["crossnobis"][vertices]))))
    return pd.DataFrame(rows)


def random_rotations(n, rng):
    """Rotation matrices for the left hemisphere and the mirrored ones for the right (x reflected)."""
    left = Rotation.random(n, random_state=rng).as_matrix()
    F = np.diag([-1.0, 1.0, 1.0])
    return left, F @ left @ F


def spin_maps(values, spheres, rotations):
    """Yield spun copies of a full vertex map (NaN outside cortex stays attached to the rotated positions).
    Each vertex takes the value of the original vertex nearest to its position after inverse rotation."""
    trees = [cKDTree(s) for s in spheres]
    n_left = len(spheres[0])
    halves = [values[:n_left], values[n_left:]]
    for rl, rr in zip(*rotations):
        out = []
        for s, tree, half, R in zip(spheres, trees, halves, (rl, rr)):
            _, nearest = tree.query(s @ R)  # rows rotated by R^-1 (R orthogonal: s @ R == (R.T @ s.T).T)
            out.append(half[nearest])
        yield np.concatenate(out)


def overlap_fraction(values, cortex, network_mask, fraction):
    """Fraction of the top `fraction` of cortical vertices (by value, ignoring NaN) that lie in the network."""
    ok = cortex & np.isfinite(values)
    vertices = np.flatnonzero(ok)
    top = top_vertices(values, vertices, fraction)
    return network_mask[top].mean()


def spin_test(values, cortex, networks, spheres, n_rotations, fraction, rng):
    """For each network mask: observed overlap of the map's top vertices with the network, its enrichment
    over the network's share of cortex, and the spin p-value. The same rotations serve all networks."""
    obs = {k: overlap_fraction(values, cortex, m, fraction) for k, m in networks.items()}
    null = {k: [] for k in networks}
    for v in spin_maps(np.where(cortex, values, np.nan), spheres, random_rotations(n_rotations, rng)):
        for k, m in networks.items():
            null[k].append(overlap_fraction(v, cortex, m, fraction))
    out = {}
    for k, m in networks.items():
        base, nl = m[cortex].mean(), np.array(null[k])
        out[k] = dict(overlap=obs[k], enrichment=obs[k] / base, null_mean_enrichment=nl.mean() / base,
                      p_spin=(1 + (nl >= obs[k]).sum()) / (1 + n_rotations))
    return out
