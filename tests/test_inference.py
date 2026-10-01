import numpy as np
import pandas as pd
import pytest

from tribeloc.inference import assign_shards, build_events, window_masks

WINDOWS = {"answer": {"anchor": "answer_onset", "start": 0.0, "end": 6.0},
           "full": {"anchor": "stimulus", "start": 0.0, "end": 0.0},
           "full_tail": {"anchor": "stimulus", "start": 0.0, "end": 4.0}}


def row(**kw):
    base = dict(stim_id="t__00000__clean_A", text="The doctors respect", onset_s=10.0, answer_onset_s=11.0, end_s=11.5)
    base.update(kw)
    return pd.DataFrame([base])


def test_assign_shards_keeps_items_whole():
    t = pd.DataFrame(dict(task=["a"] * 12 + ["b"] * 8, item=np.repeat([0, 1, 2, 0, 1], 4)))
    s = assign_shards(t, shard_size=8)
    assert list(s) == [0] * 8 + [1] * 8 + [2] * 4
    assert t.assign(s=s).groupby(["task", "item"]).s.nunique().eq(1).all()


def test_build_events_word_timing_and_context():
    e = build_events(row(), word_seconds=0.5, timeline_seconds=100)
    words = e[e.type == "Word"]
    assert list(words.start) == [10.0, 10.5, 11.0]
    assert list(words.context) == ["The", "The doctors", "The doctors respect"]
    assert (e[e.type == "Event"].duration == 100).all()


def test_window_masks_at_1hz():
    times = np.arange(100.0)
    m = window_masks(times, next(row().itertuples()), WINDOWS)
    assert list(times[m["answer"]]) == [11, 12, 13, 14, 15, 16]
    assert list(times[m["full"]]) == [10, 11]
    assert list(times[m["full_tail"]]) == [10, 11, 12, 13, 14, 15]


def test_window_masks_rejects_empty_window():
    with pytest.raises(ValueError):
        window_masks(np.arange(5.0), next(row().itertuples()), WINDOWS)
