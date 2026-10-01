import numpy as np
import pandas as pd
import pytest

from tribeloc import ROOT, load_config
from tribeloc.stimuli import (answer_word_index, build_table, filter_unique, item_stimuli,
                              join, load_tasks, split_halves)


def item(cp, kp, a, b):
    return dict(clean_problem=cp, corrupted_problem=kp, clean_correct=a, clean_incorrect=b,
                corrupted_correct=b, corrupted_incorrect=a)


def test_join_language_appends_answer_with_its_own_space():
    text, char = join("Lan", "The doctors", " respect")
    assert text == "The doctors respect"
    assert answer_word_index(text, char) == 2


def test_join_wug_fuses_answer_into_last_word():
    text, char = join("Lan", "Now there are two utetorn", "ities")
    assert text == "Now there are two utetornities"
    assert answer_word_index(text, char) == 4


def test_join_hypernymy_drops_leading_space():
    text, char = join("Lan", " diamonds, and other", " gem")
    assert text == "diamonds, and other gem"
    assert answer_word_index(text, char) == 3


def test_join_other_domains_use_one_space():
    text, char = join("MD", "858 minus 341 equals", "517")
    assert text == "858 minus 341 equals 517"
    assert answer_word_index(text, char) == 4
    text, char = join("MD", "Sort these\n## numbers\n", "['1', '2']")
    assert text == "Sort these\n## numbers ['1', '2']"
    assert answer_word_index(text, char) == 4


def test_item_stimuli_is_crossed():
    s = item_stimuli("phys", item("Put at the top?", "Put at the bottom?", "ice cream", "hamper"))
    labels = {(x["problem"], x["answer"]): x["correct"] for x in s}
    assert labels == {("clean", "A"): True, ("clean", "B"): False,
                      ("corrupted", "A"): False, ("corrupted", "B"): True}
    assert s[0]["text"] == "Put at the top? ice cream"
    assert all(x["answer_word"] == 4 for x in s)


def test_item_stimuli_rejects_uncrossed_items():
    bad = item("p", "q", "a", "b")
    bad["corrupted_correct"] = "c"
    with pytest.raises(AssertionError):
        item_stimuli("MD", bad)


def test_filter_unique_drops_exact_mirror_and_shared_items():
    items = [item("p1", "q1", "a", "b"),
             item("p1", "q1", "a", "b"),   # exact duplicate
             item("q1", "p1", "b", "a"),   # mirror: same four stimuli
             item("p1", "q2", "a", "c"),   # shares "p1 a"
             item("p3", "q3", "a", "b")]   # independent
    stimuli = [item_stimuli("MD", x) for x in items]
    assert filter_unique(stimuli) == [0, 4]


def test_build_table_filter_is_dataset_wide():
    shared = item("What prints? x", "What prints? y", "1", "2")
    tasks = [("MD", "code_list", [shared]), ("MD", "code_B", [shared, item("p", "q", "a", "b")])]
    t = build_table(tasks, word_seconds=0.5, onset_seconds=10, half_seed=0)
    assert t.groupby("task", sort=False).item.nunique().to_dict() == {"code_list": 1, "code_B": 1}


def test_filter_unique_drops_items_with_repeated_texts():
    stimuli = [item_stimuli("MD", item("p", "p", "a", "b"))]
    assert filter_unique(stimuli) == []


def test_split_halves_is_balanced_and_deterministic():
    h = split_halves(11, 1, 3)
    assert (h == 1).sum() == 5 and (h == 2).sum() == 6
    assert np.array_equal(h, split_halves(11, 1, 3))
    assert not np.array_equal(split_halves(200, 1, 3), split_halves(200, 1, 4))


def test_build_table_timing_and_halves():
    tasks = [("Lan", "sva", [item("The doctors", "The doctor", " respect", " respects"),
                             item("The dogs", "The dog", " run", " runs")])]
    t = build_table(tasks, word_seconds=0.5, onset_seconds=10, half_seed=0)
    assert len(t) == 8
    assert (t.answer_onset_s == 11.0).all() and (t.end_s == 11.5).all()
    assert t.groupby("item").half.nunique().eq(1).all()
    assert sorted(t.groupby("item").half.first()) == [1, 2]


@pytest.fixture(scope="module")
def table():
    path = ROOT / load_config()["stimuli"]["output"]
    if not path.exists():
        pytest.skip("stimulus table not built")
    return pd.read_csv(path)


def test_real_table_structure(table):
    assert table.task.nunique() == 46
    assert table.text.is_unique
    per_item = table.groupby(["task", "item"])
    assert per_item.size().eq(4).all()
    assert per_item.correct.sum().eq(2).all()
    assert per_item.half.nunique().eq(1).all()
    # every answer is correct once and incorrect once within an item
    assert table.groupby(["task", "item", "answer"]).correct.sum().eq(1).all()
    assert (table.answer_onset_s < table.end_s).all()


def test_real_table_answer_word_matches_text(table):
    words = table.text.str.split()
    assert (table.n_words == words.str.len()).all()
    # the two versions of a problem share every word before the answer
    for _, g in table.groupby(["task", "item", "problem"]):
        a, b = (w[: n] for w, n in zip(g.text.str.split(), g.answer_word))
        assert a == b
