import numpy as np
import pandas as pd

from tribeloc.rsa import (crossnobis, crossnobis_null, item_differences, item_index, lda_accuracy,
                          searchlight_accuracy, searchlight_crossnobis, whiten)


def rows_for(n_items, shuffle_rng=None):
    r = pd.DataFrame([dict(item=i, problem=p, answer=a) for i in range(n_items)
                      for p in ["clean", "corrupted"] for a in ["A", "B"]])
    if shuffle_rng is not None:
        r = r.iloc[shuffle_rng.permutation(len(r))].reset_index(drop=True)
    return r


def synthetic(n_items, v, effect, rng):
    """Stimulus patterns = item + problem + answer effects + correctness effect + noise."""
    rows = rows_for(n_items, rng)
    item = rng.normal(size=(n_items, v)) * 3
    prob = rng.normal(size=(n_items, 2, v))
    ans = rng.normal(size=(n_items, 2, v))
    direction = rng.normal(size=v)
    X = np.empty((len(rows), v))
    for k, r in enumerate(rows.itertuples()):
        p, a = int(r.problem == "corrupted"), int(r.answer == "B")
        correct = p == a
        X[k] = item[r.item] + prob[r.item, p] + ans[r.item, a] + (effect / 2 if correct else -effect / 2) * direction \
            + rng.normal(size=v) * 0.5
    return X, rows


def test_item_differences_cancel_item_problem_and_answer_effects():
    rng = np.random.default_rng(0)
    X, rows = synthetic(30, 5, effect=0.0, rng=rng)
    idx, items = item_index(rows)
    assert list(items) == list(range(30))
    assert (rows.loc[idx[:, 0], "problem"] == "clean").all() and (rows.loc[idx[:, 3], "answer"] == "B").all()
    D = item_differences(X, idx)
    # with no correctness effect, D only contains the stimulus noise (SD 0.5 * sqrt(4)/2 = 0.5)
    assert abs(D.std() - 0.5) < 0.05


def test_crossnobis_equals_mean_pairwise_inner_product():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(12, 4)) + 0.3
    G = W @ W.T
    brute = (G.sum() - np.trace(G)) / (12 * 11 * 4)
    assert np.isclose(crossnobis(W), brute)


def test_crossnobis_unbiased_under_null_and_positive_with_effect():
    rng = np.random.default_rng(2)
    null = [crossnobis(whiten(rng.normal(size=(100, 10)))) for _ in range(300)]
    assert abs(np.mean(null)) < 3 * np.std(null) / np.sqrt(300)
    shifted = crossnobis(whiten(rng.normal(size=(100, 10)) + 0.5))
    assert shifted > np.max(null)


def test_sign_flip_null_matches_recomputation():
    rng = np.random.default_rng(3)
    D = rng.normal(size=(40, 6)) + 0.2
    W = whiten(D)
    signs = rng.choice([-1.0, 1.0], size=(5, 40))
    for s, u in zip(signs, crossnobis_null(W, signs)):
        # whitening uses the uncentred second moment, which sign flips leave unchanged
        assert np.isclose(u, crossnobis(whiten(D * s[:, None])))
    assert np.isclose(crossnobis_null(W, np.ones((1, 40)))[0], crossnobis(W))


def test_searchlight_crossnobis_detects_local_effect():
    rng = np.random.default_rng(4)
    D = rng.normal(size=(200, 20))
    D[:, :5] += 0.4
    sl = (np.array([0, 5, 10, 15, 20]), np.arange(20))
    u, null = searchlight_crossnobis(D, sl, rng.choice([-1.0, 1.0], size=(200, 200)))
    assert u[0] > null[:, 0].max()
    assert u[1:].max() < null[:, 1:].max()
    assert null.shape == (200, 4)


def test_lda_accuracy_chance_and_signal():
    rng = np.random.default_rng(5)
    labels = np.array([True, False, False, True])
    for effect, lo, hi in [(0.0, 0.4, 0.6), (3.0, 0.9, 1.01)]:
        X, rows = synthetic(200, 8, effect, rng)
        idx, _ = item_index(rows)
        P = X[idx]
        Z = P - P.mean(1, keepdims=True)
        acc = lda_accuracy(Z, labels, np.arange(150), np.arange(150, 200))
        assert lo <= acc <= hi, (effect, acc)


def test_searchlight_accuracy_shapes_and_matched_n():
    rng = np.random.default_rng(6)
    X, rows = synthetic(60, 6, 3.0, rng)
    idx, _ = item_index(rows)
    sl = (np.array([0, 3, 6]), np.arange(6))
    mean, sd = searchlight_accuracy(X, idx, sl, n_items=40, n_repeats=3, n_folds=4, rng=rng)
    assert mean.shape == (2,) and (mean > 0.8).all() and (sd >= 0).all()


def test_ledoit_wolf_matches_sklearn():
    from sklearn.covariance import ledoit_wolf as sk
    from tribeloc.rsa import ledoit_wolf
    rng = np.random.default_rng(7)
    for n, p in [(484, 38), (50, 80), (200, 3)]:
        X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) + 0.3
        assert np.allclose(ledoit_wolf(X), sk(X, assume_centered=True)[0])
