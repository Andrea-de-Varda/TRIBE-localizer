"""BrainGPT literature-prior analysis: abstract construction, perplexity scoring and summary statistics.

For each task, a short abstract describes what participants read (a neutral description plus real example items,
no construct or network names) and ends with a results sentence naming one candidate set of brain regions. The
candidate with the lowest perplexity is BrainGPT's pick (Luo et al., BrainBench).
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

# Words that would name the construct or network in the hand-written descriptions (examples stay verbatim).
BANNED = ["theory of mind", "mental", "belief", "believ", "reason", "physic", "intuitive", "working memory",
          "memory", "language", "linguistic", "grammar", "syntax", "social", "moral", "emotion", "executive",
          "demand", "cognitive", "logic", "math", "arithmetic", "network", "cortex", "brain"]


def load_config():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def check_descriptions(desc):
    """Raise if any hand-written description contains a banned (construct-naming) word."""
    bad = [(t, w) for t, d in zip(desc.task, desc.description) for w in BANNED if w in d.lower()]
    if bad:
        raise ValueError(f"construct-naming words in descriptions: {bad}")


def clean(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")


def pick_examples(rows, n, max_words, rng):
    """n correct stimuli of one task: random among those with at most max_words words, else the shortest
    (truncated). Returned as quoted strings joined by spaces."""
    rows = rows[rows.correct]
    short = rows[rows.n_words <= max_words]
    pool = short if len(short) >= n else rows.nsmallest(n, "n_words")
    chosen = pool.iloc[rng.choice(len(pool), n, replace=False)] if len(pool) > n else pool
    return " ".join(f'"{clean(t, max_words)}"' for t in chosen.text)


def build_texts(cfg, desc, stimuli):
    """One row per task x candidate style x candidate x results paraphrase."""
    check_descriptions(desc)
    rng = np.random.default_rng(cfg["examples"]["seed"])
    rows = []
    for task, g in stimuli.groupby("task", sort=False):
        d = desc.set_index("task").loc[task, "description"]
        ex = pick_examples(g, cfg["examples"]["n"], cfg["examples"]["max_words"], rng)
        context = cfg["prefix"] + cfg["background"].format(description=d, examples=ex) + " "
        for style, cands in cfg["candidates"].items():
            for cand in CANDIDATES:
                for k, template in enumerate(cfg["results"]):
                    result = template.format(regions=cands[cand])
                    rows.append(dict(task=task, domain=g.domain.iloc[0], style=style, candidate=cand, paraphrase=k,
                                     context=context, result=result, text=context + result))
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

def task_preferences(scores, measure="ppl"):
    """Per task and style: preference for each candidate, averaged over paraphrases. For ppl: minus
    (log perplexity minus its mean over candidates), so higher = preferred and the mean over candidates is 0.
    For logprob_result: log-probability minus its mean over candidates."""
    s = scores.copy()
    s["value"] = -np.log(s.ppl) if measure == "ppl" else s.logprob_result
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
