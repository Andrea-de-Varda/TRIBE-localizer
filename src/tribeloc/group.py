"""Group-level statistics on vertex maps: domain contrasts, parcel-level tests and spin tests."""
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


# ── univariate parcel summaries ───────────────────────────────────────────────

def parcel_responses(task_maps, vertex_sets):
    """(n_parcels x n_tasks) mean response over each parcel's vertices."""
    return np.stack([task_maps[:, v].mean(1) for v in vertex_sets])


def target_effect(values, domains, target):
    """Mean over target-domain tasks minus the mean of the other domains' means (tasks weighted equally)."""
    others = [values[domains == d].mean() for d in DOMAINS if d != target]
    return values[domains == target].mean() - np.mean(others)


def target_permutation(values, domains, target, n_perm, rng):
    """target_effect and its one-sided p from permuting the task-to-domain labels."""
    obs = target_effect(values, domains, target)
    null = np.array([target_effect(values, rng.permutation(domains), target) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (1 + n_perm)


def pairwise_permutation(values, domains, target, other, n_perm, rng):
    """Target minus one other domain (task means), one-sided p from permuting labels between the two."""
    keep = np.isin(domains, [target, other])
    v, d = values[keep], domains[keep]
    diff = lambda lab: v[lab == target].mean() - v[lab == other].mean()
    obs = diff(d)
    null = np.array([diff(rng.permutation(d)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (1 + n_perm)


def task_selectivity(values, domains):
    """Per task: its response minus the mean of the other three domains' mean responses (tasks weighted
    equally within domain). Averaging over a domain's tasks gives target_effect for that domain."""
    means = {d: values[domains == d].mean() for d in DOMAINS}
    return np.array([v - np.mean([means[o] for o in DOMAINS if o != d]) for v, d in zip(values, domains)])


def paired_signflip(diff, n_perm, rng):
    """One-sided p for mean(diff) > 0 by flipping the sign of each paired difference; exact when 2^n <= n_perm."""
    diff = np.asarray(diff, float)
    n = len(diff)
    if 2 ** n <= n_perm:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        null = signs @ diff / n
        return diff.mean(), (null >= diff.mean() - 1e-12).mean()
    null = rng.choice([-1.0, 1.0], size=(n_perm, n)) @ diff / n
    return diff.mean(), (1 + (null >= diff.mean() - 1e-12).sum()) / (1 + n_perm)
