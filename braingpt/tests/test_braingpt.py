import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bglib import (CANDIDATES, HERE, ROOT, accuracy_permutation, build_texts, confusion, load_config,  # noqa: E402
                   pick_example, picks, score_text, task_preferences)


def test_task_specs_cover_all_tasks():
    spec = pd.read_csv(HERE / "tasks.tsv", sep="\t")
    tasks = pd.read_csv(ROOT / "data" / "stimuli" / "task_summary.csv").task
    assert set(spec.task) == set(tasks) and spec.task.is_unique
    assert spec.notna().all().all() and (spec.background.str.len() > 30).all() and (spec.manipulation.str.len() > 30).all()
    assert spec.manipulation.is_unique                      # no two tasks share a manipulation description


def fake_stimuli():
    rows = []
    for task, dom in [("t1", "Lan"), ("t2", "MD")]:
        for item in range(3):
            for problem in ["clean", "corrupted"]:
                for answer in "AB":
                    rows.append(dict(task=task, domain=dom, item=item, problem=problem, answer=answer,
                                     text=f"{task} {problem} item{item}  answer{answer}\n"))
    return pd.DataFrame(rows)


def test_pick_example_returns_the_clean_pair_of_one_item():
    s = fake_stimuli()
    item, correct, incorrect = pick_example(s[s.task == "t1"], np.random.default_rng(0))
    assert correct == f"t1 clean item{item} answerA" and incorrect == f"t1 clean item{item} answerB"


def test_abstracts_contain_the_pair_and_candidates_differ_only_in_the_regions():
    cfg = load_config()
    spec = pd.DataFrame(dict(task=["t1", "t2"], background=["Background one.", "Background two."],
                             manipulation=["Manipulation one.", "Manipulation two."],
                             correct_label=["grammatical", "correct"], incorrect_label=["ungrammatical", "incorrect"]))
    t = build_texts(cfg, spec, fake_stimuli())
    assert len(t) == 2 * len(cfg["candidates"]) * len(CANDIDATES) * len(cfg["results"])
    for (task, style, k), g in t.groupby(["task", "style", "paraphrase"]):
        ctx = g.context.iloc[0]
        assert g.context.nunique() == 1 and g.candidate.tolist() == CANDIDATES
        item = g.example_item.iloc[0]
        assert f'"{task} clean item{item} answerA"' in ctx and f'"{task} clean item{item} answerB"' in ctx
        stripped = {r.result.replace(cfg["candidates"][style][r.candidate], "{regions}") for r in g.itertuples()}
        assert stripped == {cfg["results"][k]}
    assert t.text.str.startswith(cfg["prefix"]).all()


class CharTokenizer:
    """One token per character; offsets are character positions."""
    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        import torch
        ids = torch.tensor([[ord(c) % 50 for c in text]])
        offs = torch.tensor([[(i, i + 1) for i in range(len(text))]])
        return type("Enc", (), dict(input_ids=ids, offset_mapping=offs))()


class UniformModel:
    device = "cpu"

    def __call__(self, ids):
        import torch
        return type("Out", (), dict(logits=torch.zeros(1, ids.shape[1], 50)))()


def test_score_text_attributes_tokens_to_the_result_sentence():
    pytest.importorskip("torch")
    r = score_text(UniformModel(), CharTokenizer(), "context ", "result.")
    assert r["n_tokens"] == len("context result.") - 1
    assert r["n_result_tokens"] == len("result.")
    assert r["ppl"] == pytest.approx(50.0)
    assert r["logprob_result"] == pytest.approx(-len("result.") * np.log(50))


def synthetic_scores(correct=True):
    rows = []
    for t in range(20):
        dom = ["Lan", "MD", "phys", "ToM"][t % 4]
        for c in CANDIDATES:
            for k in range(3):
                best = (c == dom) if correct else (c == "visual")
                rows.append(dict(task=f"t{t}", domain=dom, style="anatomical", candidate=c, paraphrase=k,
                                 ppl=10.0 if best else 12.0, logprob_result=-5.0 if best else -6.0))
    return pd.DataFrame(rows)


def test_analysis_perfect_and_constant_pickers():
    rng = np.random.default_rng(0)
    p = picks(task_preferences(synthetic_scores(True)))
    acc, pv = accuracy_permutation(p, 2000, rng)
    assert acc == 1.0 and pv < .001
    assert np.allclose(task_preferences(synthetic_scores(True))[CANDIDATES].sum(1), 0)
    C = confusion(p)
    assert (np.diag(C.to_numpy()[:, :4]) == 5).all()
    p = picks(task_preferences(synthetic_scores(False), "logprob_result"))
    acc, pv = accuracy_permutation(p, 2000, rng)
    assert acc == 0.0 and pv == 1.0 and (p.pick == "visual").all()
