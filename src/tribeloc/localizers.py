"""Text localizer stimuli: language (sentences vs nonwords), MD (hard vs easy arithmetic), ToM (false belief vs
false photograph) and two text adaptations of the TowerLoc physics localizer.

Every localizer produces rows with: localizer, condition, pair (stimuli sharing a source item; the unit of the
split halves and of paired tests), source_id, text. `build_table` adds timing and ids in the format used by
inference (one isolated timeline per stimulus).
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd

WORD = re.compile(r"\S+")

# ── language: EvLab sentences vs nonwords (all 10 run/set CSVs) ──────────────────────────────────────────

def language_rows(zip_bytes):
    """Sentences (S) and nonword strings (N), 12 words each, lowercased as in AlKhamissi et al. (2025)."""
    outer = zipfile.ZipFile(io.BytesIO(zip_bytes))
    rows = []
    for name in sorted(n for n in outer.namelist() if re.fullmatch(r"langloc_fmri_run\d_stim_set\d\.csv", n)):
        d = pd.read_csv(outer.open(name))
        for r in d.itertuples(index=False):
            words = [str(getattr(r, f"stim{i}")).strip().lower() for i in range(2, 14)]
            assert len(words) == 12 and all(len(w.split()) == 1 for w in words), (name, words)
            cond = {"S": "sentence", "N": "nonword"}[r.stim14]
            rows.append(dict(localizer="language", condition=cond, source_id=f"{name[:-4]}_{r.stim1}", text=" ".join(words)))
    rows = pd.DataFrame(rows)
    rows["pair"] = rows.groupby("condition").cumcount()          # no natural pairing; index within condition
    return rows


# ── ToM: Saxe lab false belief vs false photograph (Dodell-Feder et al., 2011) ───────────────────────────

def _clean(text):
    text = re.split(r"\bTrue\s+False\b", text, flags=re.I)[0]
    text = re.sub(r"(\w)-[ \t]*\r?\n\s*(\w)", r"\1-\2", text)      # hyphenated word split across lines
    return " ".join(text.split())


def tom_rows(zip_bytes):
    """Story followed by its question (no answer), 10 belief and 10 photo items."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    read = lambda n: _clean(z.read(f"tomloc/{n}").decode("utf-8", errors="replace"))
    rows = []
    for cond, tag in [("belief", "b"), ("photo", "p")]:
        for i in range(1, 11):
            text = read(f"{i}{tag}_story.txt") + " " + read(f"{i}{tag}_question.txt")
            rows.append(dict(localizer="tom", condition=cond, source_id=f"{i}{tag}", pair=i - 1, text=text))
    return pd.DataFrame(rows)


# ── MD: hard vs easy arithmetic, exactly as in AlKhamissi et al. (2025), llm-localization MDLocDataset ────

def md_rows(n=100, seed=42):
    """Hard: operands 100-199; easy: operands 1-19; addition or subtraction; the answer is included.
    Reproduces MDLocDataset (both conditions re-seeded with the same seed), then drops exact duplicate
    problems (easy problems repeat), because identical texts get identical predictions."""
    rows = []
    for cond, (lo, hi) in [("hard", (100, 200)), ("easy", (1, 20))]:
        np.random.seed(seed)
        for i in range(n):
            a, b = np.random.randint(lo, hi), np.random.randint(lo, hi)
            op = np.random.choice(["+", "-"])
            ans = a + b if op == "+" else a - b
            rows.append(dict(localizer="md", condition=cond, source_id=f"{cond}_{i:03d}", pair=i,
                             text=f"Question: Solve {a} {op} {b}?\nAnswer: {ans}"))
    return pd.DataFrame(rows).drop_duplicates("text", keep="first").reset_index(drop=True)


# ── physics: two text adaptations of TowerLoc (Fischer et al., 2016) ─────────────────────────────────────
# Cues are the TowerLoc cues used in the visual pilot; each stimulus starts with its cue, as the task cue
# precedes the tower in TowerLoc. Every block sentence has a fixed number of words, so matched versions of a
# tower have exactly the same length.

PHYSICS_CUE, COLOUR_CUE = "Where will it fall?", "More blue or yellow?"
ORDINALS = ["bottom", "second", "third", "fourth", "fifth", "sixth"]
NUMBERS = {4: "four", 5: "five", 6: "six"}
SHAPES = ["cube", "block", "slab", "brick", "box"]
WEIGHTS = ["heavy", "light"]
COLOURS = ["blue", "yellow"]
BASE = "resting flat on the table"                                           # 5 words
PLACEMENTS = ["set far to the left", "set far to the right", "set slightly to the left",
              "set slightly to the right", "set right in the middle"]          # 5 words each
PATTERNS = ["painted with a yellow stripe", "painted with a blue stripe", "covered in small yellow dots",
            "covered in small blue dots", "with one bright yellow face", "with one bright blue face"]   # 5 words each


def make_tower(rng):
    n = int(rng.integers(4, 7))
    blocks = []
    for i in range(n):
        blocks.append(dict(ordinal=ORDINALS[i], shape=str(rng.choice(SHAPES)), weight=str(rng.choice(WEIGHTS)),
                           colour=str(rng.choice(COLOURS)), pattern=str(rng.choice(PATTERNS)),
                           placement=BASE if i == 0 else str(rng.choice(PLACEMENTS))))
    return blocks


def describe(blocks, kind):
    """kind: 'both' (standard: weight, colour, placement and pattern), 'physical' (weight + placement only) or
    'colour' (colour + pattern only). 'physical' and 'colour' sentences have identical word counts."""
    parts = [f"A tower of {NUMBERS[len(blocks)]} blocks stands on a table."]
    for b in blocks:
        head = f"The {b['ordinal']} block is a"
        if kind == "both":
            parts.append(f"{head} {b['weight']} {b['colour']} {b['shape']} {b['placement']}, {b['pattern']}.")
        elif kind == "physical":
            parts.append(f"{head} {b['weight']} {b['shape']} {b['placement']}.")
        elif kind == "colour":
            parts.append(f"{head} {b['colour']} {b['shape']} {b['pattern']}.")
        else:
            raise ValueError(kind)
    return " ".join(parts)


def physics_rows(n_towers=60, seed=20261006):
    """physics_task (standard TowerLoc logic): the same description of a tower, preceded by the physics or the
    colour cue; conditions differ only in the cue. physics_content: the physical description (weights,
    placements) with the physics cue vs the colour description (colours, patterns) of the same tower with the
    colour cue; matched in length word for word."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_towers):
        blocks = make_tower(rng)
        both = describe(blocks, "both")
        rows += [dict(localizer="physics_task", condition="physics", source_id=f"tower_{t:02d}", pair=t, text=f"{PHYSICS_CUE} {both}"),
                 dict(localizer="physics_task", condition="colour", source_id=f"tower_{t:02d}", pair=t, text=f"{COLOUR_CUE} {both}"),
                 dict(localizer="physics_content", condition="physics", source_id=f"tower_{t:02d}", pair=t,
                      text=f"{PHYSICS_CUE} {describe(blocks, 'physical')}"),
                 dict(localizer="physics_content", condition="colour", source_id=f"tower_{t:02d}", pair=t,
                      text=f"{COLOUR_CUE} {describe(blocks, 'colour')}")]
    return pd.DataFrame(rows)


# ── table ─────────────────────────────────────────────────────────────────────────────────────────────────

def assign_halves(rows, seed):
    """Split pairs of each localizer into two random halves (both conditions of a pair stay together)."""
    rows = rows.copy()
    rows["half"] = 0
    for k, (loc, g) in enumerate(rows.groupby("localizer", sort=False)):
        pairs = np.sort(g.pair.unique())
        perm = np.random.default_rng([seed, k]).permutation(pairs)
        h1 = set(perm[: len(pairs) // 2].tolist())
        rows.loc[g.index, "half"] = np.where(g.pair.isin(h1), 1, 2)
    return rows


def build_table(rows, word_seconds, onset_seconds, half_seed):
    rows = assign_halves(rows.reset_index(drop=True), half_seed)
    rows["n_words"] = [len(WORD.findall(t)) for t in rows.text]
    rows["onset_s"] = onset_seconds
    rows["end_s"] = onset_seconds + rows.n_words * word_seconds
    rows["answer_onset_s"] = rows.onset_s            # not used; keeps the inference row format
    rows["task"] = rows.localizer                    # shard key (with item)
    rows["item"] = np.arange(len(rows))
    rows["stim_id"] = [f"loc_{r.localizer}__{r.condition}__{r.pair:03d}" for r in rows.itertuples()]
    cols = ["stim_id", "localizer", "condition", "pair", "half", "source_id", "n_words", "onset_s", "end_s",
            "answer_onset_s", "task", "item", "text"]
    return rows[cols]
