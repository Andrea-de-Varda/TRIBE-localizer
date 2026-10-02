"""BrainGPT literature-prior analysis: abstract construction, perplexity scoring and summary statistics.

For each task, an abstract states what the task studies, the manipulation (what makes an item correct or incorrect)
and one sampled item pair from the original dataset, and ends with a results sentence naming one candidate set of
brain regions. The candidate with the lowest perplexity is BrainGPT's pick (Luo et al., BrainBench).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOMAINS = ["Lan", "MD", "phys", "ToM"]
CANDIDATES = DOMAINS + ["visual"]
NULL_TASK = "__neutral__"     # neutral abstract used to calibrate the results sentences (PMI)

def load_config():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def flat(text):
    return " ".join(str(text).split())


def pick_example(rows, rng):
    """One item of a task: the clean problem with its correct (answer A) and incorrect (answer B) completion."""
    item = int(rng.choice(np.sort(rows.item.unique())))
    clean = rows[(rows.item == item) & (rows.problem == "clean")].set_index("answer").text
    return item, flat(clean["A"]), flat(clean["B"])


def build_texts(cfg, tasks, stimuli):
    """One row per task x candidate style x candidate x results paraphrase."""
    rng = np.random.default_rng(cfg["examples"]["seed"])
    spec = tasks.set_index("task")
    rows = []
    for task, g in stimuli.groupby("task", sort=False):
        t = spec.loc[task]
        item, correct, incorrect = pick_example(g, rng)
        context = cfg["prefix"] + cfg["background"].format(
            background=t.background, manipulation=t.manipulation, correct_label=t.correct_label,
            incorrect_label=t.incorrect_label, correct=correct, incorrect=incorrect) + " "
        for style, cands in cfg["candidates"].items():
            for cand in CANDIDATES:
                for k, template in enumerate(cfg["results"]):
                    result = template.format(regions=cands[cand])
                    rows.append(dict(task=task, domain=g.domain.iloc[0], style=style, candidate=cand, paraphrase=k,
                                     example_item=item, context=context, result=result,
                                     text=context + result))
    null = cfg["prefix"] + cfg["null_background"] + " "
    for style, cands in cfg["candidates"].items():
        for cand in CANDIDATES:
            for k, template in enumerate(cfg["results"]):
                result = template.format(regions=cands[cand])
                rows.append(dict(task=NULL_TASK, domain="none", style=style, candidate=cand, paraphrase=k,
                                 example_item=-1, context=null, result=result, text=null + result))
    return pd.DataFrame(rows)


def score_text(model, tok, context, result):
    """Full-text perplexity (BrainBench criterion) and the summed log-probability of the results sentence given the
    context. Tokens are attributed to the results sentence by character offsets."""
    import torch
    text = context + result
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc.input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
    starts = enc.offset_mapping[0, 1:, 0].numpy()
    in_result = starts >= len(context)
    return dict(ppl=float(np.exp(-logp.mean())), logprob_result=float(logp[in_result].sum()),
                n_tokens=int(len(logp)), n_result_tokens=int(in_result.sum()))


# ── analysis ──────────────────────────────────────────────────────────────────────────────────────────────

def add_pmi(scores):
    """PMI = log p(result | task abstract) - log p(result | neutral abstract), for the same style, candidate and
    paraphrase (domain-conditional PMI; Holtzman et al., 2021). Same tokens in both terms, so phrase length and
    baseline frequency cancel. Returns the task rows only, with a `pmi` column."""
    null = scores[scores.task == NULL_TASK].set_index(["style", "candidate", "paraphrase"]).logprob_result
    s = scores[scores.task != NULL_TASK].copy()
    s["pmi"] = s.logprob_result.to_numpy() - null.loc[list(zip(s["style"], s.candidate, s.paraphrase))].to_numpy()
    return s


def task_preferences(scores, measure="pmi"):
    """Per task and style: preference for each candidate, averaged over paraphrases, minus its mean over candidates
    (higher = preferred; 0 = average). measure: pmi (primary), ppl (BrainBench full-text perplexity, as minus log
    perplexity) or logprob_result (uncalibrated results-sentence log-probability)."""
    s = add_pmi(scores) if measure == "pmi" else scores[scores.task != NULL_TASK].copy()
    s["value"] = {"pmi": lambda: s.pmi, "ppl": lambda: -np.log(s.ppl), "logprob_result": lambda: s.logprob_result}[measure]()
    m = s.groupby(["task", "domain", "style", "candidate"], sort=False).value.mean().unstack("candidate")[CANDIDATES]
    return m.sub(m.mean(1), axis=0).reset_index()


def picks(pref):
    p = pref.copy()
    p["pick"] = p[CANDIDATES].idxmax(1)
    p["correct"] = p.pick == p.domain
    return p


def confusion(p):
    """Rows: task domain; columns: picked candidate; entries: number of tasks."""
    return pd.crosstab(p.domain, p.pick).reindex(index=DOMAINS, columns=CANDIDATES, fill_value=0)


def accuracy_permutation(p, n_perm, rng):
    """Accuracy over tasks and a one-sided p from permuting task-to-domain labels (keeps the model's picks and
    the number of tasks per domain, so a model that always picks one candidate is not rewarded)."""
    acc = p.correct.mean()
    picks_, dom = p.pick.to_numpy(), p.domain.to_numpy()
    null = np.array([(picks_ == rng.permutation(dom)).mean() for _ in range(n_perm)])
    return acc, (1 + (null >= acc).sum()) / (1 + n_perm)


def balanced_accuracy(p):
    return float(np.mean([p[p.domain == d].correct.mean() for d in DOMAINS]))
