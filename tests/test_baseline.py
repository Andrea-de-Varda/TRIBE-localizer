import numpy as np
import pandas as pd

from tribeloc.baseline import group_baselines, stimulus_baselines

WINDOWS = {"full": {"anchor": "stimulus", "start": 0.0, "end": 0.0},
           "full_tail": {"anchor": "stimulus", "start": 0.0, "end": 4.0}}


def rows(ends):
    return pd.DataFrame(dict(stim_id=[f"s{i}" for i in range(len(ends))], onset_s=10.0, end_s=ends,
                             answer_onset_s=10.0))


def test_stimulus_baseline_is_the_window_mean_of_the_time_course():
    times = np.arange(100.0)
    tc = np.stack([times, 2 * times])                       # 2 vertices, value = time
    r = rows([12.0, 15.5])
    b = stimulus_baselines(r, tc, times, WINDOWS)
    assert np.allclose(b["full"][0], [10.5, 21.0])          # samples 10, 11
    assert np.allclose(b["full"][1], [12.5, 25.0])          # samples 10..15
    assert np.allclose(b["full_tail"][0], [12.5, 25.0])     # samples 10..15 (end 12 + 4)


def test_group_baselines_equal_mean_of_stimulus_baselines():
    rng = np.random.default_rng(0)
    times = np.arange(100.0)
    tc = rng.normal(size=(5, 100))
    r = rows(list(10 + rng.integers(1, 40, 30) * .5))
    groups = rng.integers(0, 3, 30)
    per = stimulus_baselines(r, tc, times, WINDOWS)
    grp = group_baselines(r, groups, 3, tc, times, WINDOWS)
    for w in WINDOWS:
        for g in range(3):
            assert np.allclose(grp[w][g], per[w][groups == g].mean(0))
