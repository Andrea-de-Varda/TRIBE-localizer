import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tribeloc.froi import contrast_t, define_frois, heldout_effect, localizer_t
from tribeloc.localizers import (COLOUR_CUE, PHYSICS_CUE, WORD, assign_halves, build_table, describe, make_tower,
                                 md_rows, physics_rows)


def n_words(t):
    return len(WORD.findall(t))


def test_md_matches_alkhamissi_and_is_deduplicated():
    r = md_rows(100, 42)
    assert r.text.is_unique
    assert (r.groupby("condition").size() <= 100).all() and (r.condition == "hard").sum() == 100
    assert set(r.text.map(n_words)) == {7}
    first = r[r.condition == "hard"].text.iloc[0]
    assert first.startswith("Question: Solve ") and "\nAnswer: " in first
    a, op, b = first.split("Solve ")[1].split("?")[0].split()
    ans = int(first.split("Answer: ")[1])
    assert ans == (int(a) + int(b) if op == "+" else int(a) - int(b))
    assert all(100 <= int(x) < 200 for x in (a, b))


def test_physics_task_differs_only_in_the_cue():
    r = physics_rows(20, 1)
    t = r[r.localizer == "physics_task"]
    for _, g in t.groupby("pair"):
        p, c = g.set_index("condition").text[["physics", "colour"]]
        assert p.startswith(PHYSICS_CUE) and c.startswith(COLOUR_CUE)
        assert p[len(PHYSICS_CUE):] == c[len(COLOUR_CUE):]
        assert n_words(p) == n_words(c)


def test_physics_content_matched_length_and_content_split():
    r = physics_rows(20, 1)
    t = r[r.localizer == "physics_content"]
    for _, g in t.groupby("pair"):
        p, c = g.set_index("condition").text[["physics", "colour"]]
        assert n_words(p) == n_words(c)
        body_p, body_c = p.split("?", 1)[1], c.split("?", 1)[1]
        assert "heavy" not in body_c and "light" not in body_c and "set " not in body_c
        assert not any(w in body_p for w in ["blue", "yellow", "stripe", "dots", "face"])


def test_describe_sentence_lengths():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = make_tower(rng)
        assert n_words(describe(b, "physical")) == n_words(describe(b, "colour")) == 9 + 12 * len(b)
        assert n_words(describe(b, "both")) == 9 + 18 * len(b)


def test_halves_keep_pairs_together_and_ids_unique():
    rows = pd.concat([md_rows(20, 42), physics_rows(10, 2)], ignore_index=True)
    t = build_table(rows, .5, 10., 7)
    assert t.stim_id.is_unique
    assert (t.groupby(["localizer", "pair"]).half.nunique() == 1).all()
    for loc, g in t.groupby("localizer"):
        n = g.pair.nunique()
        assert set(g.half) == {1, 2} and abs((g.groupby("pair").half.first() == 1).sum() - n / 2) <= 1


def test_contrast_t_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, (30, 5)), rng.normal(0, 2, (25, 5))
    assert np.allclose(contrast_t(a, b, "welch"), stats.ttest_ind(a, b, equal_var=False).statistic)
    c = rng.normal(0, 1, (30, 5))
    pairs = np.arange(30)
    perm = rng.permutation(30)
    t = contrast_t(a[perm], c, "paired", pairs[perm], pairs)
    assert np.allclose(t, stats.ttest_rel(a, c).statistic)


def test_define_frois_and_heldout_effect_recover_planted_vertices():
    rng = np.random.default_rng(1)
    n_pairs, V = 40, 50
    rows = pd.DataFrame(dict(condition=["physics"] * n_pairs + ["colour"] * n_pairs,
                             pair=list(range(n_pairs)) * 2, localizer="x"))
    rows = assign_halves(rows, 3)
    X = rng.normal(0, 1, (2 * n_pairs, V))
    X[:n_pairs, 10:15] += 2.0                                  # target > control at vertices 10-14
    spec = dict(target="physics", control="colour", test="paired")
    t = localizer_t(X, rows, spec)
    labels = np.zeros(V, int)
    labels[:25], labels[25:] = 1, 2
    fr = define_frois(t, labels, np.ones(V, bool), [1, 2], 0.2)
    assert set(fr[1]) == set(range(10, 15)) and len(fr[2]) == 5
    eff, t_ho = heldout_effect(X, rows, spec, np.arange(25), 0.2)
    assert eff == pytest.approx(2.0, abs=.4) and t_ho > 5
    eff0, _ = heldout_effect(X, rows, spec, np.arange(25, 50), 0.2)
    assert abs(eff0) < .5
