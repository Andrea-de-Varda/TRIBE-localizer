"""Searchlight crossnobis distance and LDA classification of correct vs incorrect stimuli.

For item i, with versions cA, cB (clean problem + answer A/B) and kA, kB (corrupted problem),
the correct-minus-incorrect difference is

    d_i = (x_cA + x_kB - x_cB - x_kA) / 2

which removes item identity, problem identity and answer identity. Within a searchlight the
differences are whitened by the Ledoit-Wolf-shrunk second moment M = D'D / n (uncentred, so it is
unchanged by sign flips), and the crossnobis distance is the mean inner product over all pairs of
distinct items, per vertex:

    U = (||sum_i w_i||^2 - sum_i ||w_i||^2) / (n (n - 1) V)

This is the leave-one-item-out cross-validated Mahalanobis distance: expected value 0 when there is
no correct/incorrect difference. Swapping the labels within an item flips the sign of d_i exactly,
so the permutation null uses random per-item sign flips.
"""
import numpy as np

VERSIONS = ["clean_A", "clean_B", "corrupted_A", "corrupted_B"]


def item_index(rows):
    """(n_items x 4) row positions in `rows`, columns in VERSIONS order, items sorted."""
    version = rows.problem + "_" + rows.answer
    pivot = rows.assign(pos=np.arange(len(rows)), version=version).pivot(index="item", columns="version", values="pos")
    return pivot[VERSIONS].to_numpy(), pivot.index.to_numpy()


def item_differences(X, idx):
    """d_i for every item: X (n_stimuli x V), idx from item_index."""
    return (X[idx[:, 0]] + X[idx[:, 3]] - X[idx[:, 1]] - X[idx[:, 2]]) / 2


def ledoit_wolf(X):
    """Ledoit-Wolf shrinkage of the uncentred second moment X'X/n towards a scaled identity.
    Same estimator as sklearn.covariance.ledoit_wolf(X, assume_centered=True), without its
    per-call overhead (this runs millions of times on small matrices)."""
    n, p = X.shape
    S = X.T @ X / n
    X2 = X ** 2
    trace = X2.sum() / n
    mu = trace / p
    delta_ = (S ** 2).sum()
    beta_ = (X2.sum(1) ** 2).sum() / n          # == (X2.T @ X2).sum() / n
    beta = (beta_ - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace + p * mu ** 2) / p
    shrinkage = 0.0 if delta == 0 else min(beta, delta) / delta
    S *= 1 - shrinkage
    S[np.diag_indices(p)] += shrinkage * mu
    return S


def whiten(D):
    """D @ M^(-1/2), with M the Ledoit-Wolf-shrunk uncentred second moment of the rows of D."""
    M = ledoit_wolf(D)
    vals, vecs = np.linalg.eigh(M)
    return D @ (vecs / np.sqrt(np.maximum(vals, 1e-12 * vals.max()))) @ vecs.T


def crossnobis(W):
    n, v = W.shape
    s = W.sum(0)
    return (s @ s - (W ** 2).sum()) / (n * (n - 1) * v)


def crossnobis_null(W, signs):
    """Crossnobis for each row of `signs` (n_perm x n_items, entries +-1)."""
    n, v = W.shape
    t = signs @ W
    return ((t ** 2).sum(1) - (W ** 2).sum()) / (n * (n - 1) * v)


def searchlight_crossnobis(D, sl, signs=None):
    """Crossnobis per searchlight. D: items x vertices; sl: (indptr, indices).
    Returns (n_centres,) and, if signs is given, (n_perm x n_centres) null maps."""
    indptr, indices = sl
    n_c = len(indptr) - 1
    out = np.empty(n_c)
    null = None if signs is None else np.empty((len(signs), n_c), np.float32)
    for c in range(n_c):
        W = whiten(D[:, indices[indptr[c]:indptr[c + 1]]])
        out[c] = crossnobis(W)
        if signs is not None:
            null[:, c] = crossnobis_null(W, signs)
    return out, null


def lda_accuracy(Z, labels, train, test):
    """Shrinkage-LDA accuracy for correct (True) vs incorrect stimuli.

    Z: item-centred patterns (n_items x 4 x V); labels: (4,) booleans in VERSIONS order;
    train/test: item indices. With item centring the two class means are +-dbar/2, so the
    discriminant has no bias term.
    """
    sign = np.where(labels, 1.0, -1.0)
    Zt = Z[train]
    dbar = np.einsum("ijk,j->k", Zt, sign) / (2 * len(Zt))      # mean correct-minus-incorrect difference
    resid = Zt - sign[None, :, None] * dbar / 2
    S = ledoit_wolf(resid.reshape(-1, Z.shape[2]))
    w = np.linalg.solve(S, dbar)
    pred = Z[test] @ w > 0
    return (pred == labels[None, :]).mean()


def searchlight_accuracy(X, idx, sl, n_items, n_repeats, n_folds, rng):
    """Mean and SD (over repeats) of cross-validated LDA accuracy per searchlight.
    Each repeat subsamples n_items items; folds split items, so an item's four stimuli stay together."""
    labels = np.array([True, False, False, True])                 # VERSIONS: cA, cB, kA, kB
    indptr, indices = sl
    n_c = len(indptr) - 1
    acc = np.zeros((n_repeats, n_c))
    for r in range(n_repeats):
        items = rng.choice(len(idx), min(n_items, len(idx)), replace=False)
        P = X[idx[items]]                                         # items x 4 x V
        Z = P - P.mean(1, keepdims=True)
        folds = np.array_split(rng.permutation(len(items)), n_folds)
        splits = [(np.setdiff1d(np.arange(len(items)), f), f) for f in folds]
        for c in range(n_c):
            Zc = Z[:, :, indices[indptr[c]:indptr[c + 1]]]
            acc[r, c] = np.mean([lda_accuracy(Zc, labels, tr, te) for tr, te in splits])
    return acc.mean(0), acc.std(0)
