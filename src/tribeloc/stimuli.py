"""Build the crossed 2x2 stimulus table from the LLM_Modularity items.

Each item has a clean and a corrupted problem and two answers that swap roles
(clean_correct == corrupted_incorrect, clean_incorrect == corrupted_correct).
Answer "A" is the clean-correct string and answer "B" the clean-incorrect one,
so the four stimuli of an item are:

    clean problem     + A -> correct      clean problem     + B -> incorrect
    corrupted problem + A -> incorrect    corrupted problem + B -> correct
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONS = [("clean", "A"), ("clean", "B"), ("corrupted", "A"), ("corrupted", "B")]
WORD = re.compile(r"\S+")


def load_tasks(source_dir, domains):
    """Return [(domain, task, items)] in config order. Task names are the config keys."""
    source_dir = Path(source_dir)
    tasks = []
    for domain in domains:
        config = json.loads((source_dir / "config" / domain / "config.json").read_text())
        for task, spec in config.items():
            items = json.loads((source_dir / "data" / domain / spec["path"]).read_text())
            tasks.append((domain, task, items))
    return tasks


def join(domain, problem, answer):
    """Text shown to TRIBE and the character offset where the answer starts.

    Language items were scored by direct concatenation (answers carry their own
    leading space, wug answers are suffixes of the last word). For all other
    domains the answer was a separate chat turn, so it is joined with one space.
    """
    if domain == "Lan":
        prefix = problem
    else:
        prefix = problem.rstrip() + " "
        answer = answer.strip()
    text = prefix + answer
    answer_char = len(prefix) + len(answer) - len(answer.lstrip())
    # Some problems start with whitespace (e.g. hypernymy); drop it from the text.
    lead = len(text) - len(text.lstrip())
    return text[lead:].rstrip(), answer_char - lead


def answer_word_index(text, answer_char):
    """Index of the whitespace word containing the first answer character."""
    for i, match in enumerate(WORD.finditer(text)):
        if match.start() <= answer_char < match.end():
            return i
    raise ValueError(f"answer offset {answer_char} not inside a word of {text!r}")


def item_stimuli(domain, item):
    """The four (problem, answer, correct, text, answer_word) stimuli of one item."""
    assert item["clean_correct"] == item["corrupted_incorrect"], "answers are not crossed"
    assert item["clean_incorrect"] == item["corrupted_correct"], "answers are not crossed"
    answers = {"A": item["clean_correct"], "B": item["clean_incorrect"]}
    out = []
    for problem, answer in VERSIONS:
        text, answer_char = join(domain, item[f"{problem}_problem"], answers[answer])
        correct = (problem == "clean") == (answer == "A")
        out.append(dict(problem=problem, answer=answer, correct=correct, text=text,
                        answer_word=answer_word_index(text, answer_char)))
    return out


def filter_unique(stimuli_per_item, seen=None):
    """Keep an item only if its four texts are distinct and none occurred in an
    earlier kept item. Returns indices of kept items, in source order. Passing
    the same `seen` set across tasks makes the filter dataset-wide."""
    seen = set() if seen is None else seen
    kept = []
    for i, stimuli in enumerate(stimuli_per_item):
        texts = {s["text"] for s in stimuli}
        if len(texts) == 4 and not texts & seen:
            seen |= texts
            kept.append(i)
    return kept


def split_halves(n, seed, task_index):
    """Random half assignment (1 or 2) of n items; the first floor(n/2) of a permutation get half 1."""
    order = np.random.default_rng([seed, task_index]).permutation(n)
    half = np.full(n, 2)
    half[order[: n // 2]] = 1
    return half


def build_table(tasks, word_seconds, onset_seconds, half_seed):
    rows, seen = [], set()
    for task_index, (domain, task, items) in enumerate(tasks):
        stimuli = [item_stimuli(domain, item) for item in items]
        kept = filter_unique(stimuli, seen)
        halves = split_halves(len(kept), half_seed, task_index)
        for item, (source_index, half) in enumerate(zip(kept, halves)):
            for s in stimuli[source_index]:
                n_words = len(WORD.findall(s["text"]))
                rows.append(dict(
                    stim_id=f"{task}__{item:05d}__{s['problem']}_{s['answer']}",
                    domain=domain, task=task, task_index=task_index, item=item,
                    source_index=source_index, half=int(half),
                    problem=s["problem"], answer=s["answer"], correct=s["correct"],
                    n_words=n_words, answer_word=s["answer_word"],
                    onset_s=onset_seconds,
                    answer_onset_s=onset_seconds + s["answer_word"] * word_seconds,
                    end_s=onset_seconds + n_words * word_seconds,
                    text=s["text"]))
    return pd.DataFrame(rows)


def summarize(tasks, table):
    """Per-task counts before and after the uniqueness filter."""
    raw = pd.Series({task: len(items) for _, task, items in tasks}, name="n_source")
    kept = table.groupby("task", sort=False).item.nunique().rename("n_items")
    words = table.groupby("task", sort=False).n_words.mean().rename("mean_words")
    domain = table.groupby("task", sort=False).domain.first()
    out = pd.concat([domain, raw, kept, words], axis=1).loc[raw.index]
    out["n_removed"] = out.n_source - out.n_items
    return out.reset_index(names="task")
