# BrainGPT literature prior for the 46 tasks

A separate, complementary analysis to the TRIBE localization in the parent repository. TRIBE asks where text drives predicted brain responses. Here we ask which brain network the neuroscience literature would expect a study of each task to report, using BrainGPT (a LoRA adapter for Mistral-7B-v0.1 trained on PubMed Central neuroscience papers, 2002–2022; Luo et al., Nature Human Behaviour; BrainBench). BrainGPT is a scorer, not a predictor of activation: for an abstract with several candidate results, the candidate with the lowest perplexity is the one the model finds most plausible, and the perplexity gap is its confidence. This is evidence about the literature's expectations (a stand-in for expert consensus), not a measurement.

## Layout

```
config.yaml           model, prompt prefix, abstract template, result paraphrases, candidate regions
tasks.tsv             hand-written per task: background (what it studies), manipulation (what makes an item correct or not), labels
bglib.py              abstract construction, scoring, analysis helpers
build_abstracts.py    -> data/abstracts.csv (all texts to score), data/abstracts_preview.txt (one abstract per task)
score.py              -> results/scores.csv (GPU)
analyze.py            -> results/{preferences,picks,summary}.csv, plots/
slurm/score.sbatch    scoring + analysis on Engaging
tests/                pytest (run from the repository root: python -m pytest)
```

## How to run

```bash
python braingpt/build_abstracts.py            # local
# on Engaging, from the repository root; once: pip install peft accelerate --no-deps (tribe env) and HF access to mistralai/Mistral-7B-v0.1
sbatch braingpt/slurm/score.sbatch            # 1 GPU: score.py, then analyze.py
```

## Decision log

### 2026-10-01 — Design (decisions: Andrea)

**Abstracts** (redesigned 2026-10-01 after Andrea's review; see the end of this log). One per task, written like a real abstract: what the task studies, naming the construct (decision: Andrea: the construct is what we want the literature prior to respond to); the experimental manipulation, i.e. what makes an item correct or incorrect (e.g. grammatical vs ungrammatical for agreement, physically plausible vs implausible for the physics tasks); and one item sampled from the original dataset, shown in its correct and its incorrect version (the clean problem with its correct and its incorrect answer, seed 20261008, verbatim, whitespace collapsed). Template: "{background} We used fMRI to measure brain responses while healthy adult participants read short written items, presented one word at a time, and judged whether each item was {correct label}. {manipulation} For example, "{correct item}" ({correct label}) versus "{incorrect item}" ({incorrect label}). Items were presented in blocks, and responses were compared with a fixation baseline." Background, manipulation and labels for every task are in `tasks.tsv`; every task has its own manipulation description (the physics tasks name their property: buoyancy, light absorption, elasticity, solubility, speed, stability, temperature). The abstract is identical across candidates, as BrainBench performance depends on the background and methods. Prompt prefix as in BrainBench scoring: "You are a neuroscientist with deep knowledge in neuroscience. Here is an abstract from a neuroscience publication: ".

**Candidates.** Five per abstract: the four target networks and early visual cortex as a distractor that no task targets (decision: Andrea). Two phrasing styles, both analysed (decision: Andrea):
- anatomical: Language "the left inferior frontal gyrus and the left lateral temporal cortex"; MD "the bilateral middle frontal gyrus, precentral sulcus and intraparietal sulcus"; ToM "the bilateral temporo-parietal junction, precuneus and medial prefrontal cortex"; Physics "the bilateral dorsal premotor cortex, supplementary motor area and superior parietal lobule"; visual "the bilateral calcarine sulcus and occipital pole". The physics regions are described as distinctly from MD as the literature allows; the two systems overlap anatomically.
- network names: "the language network", "the multiple-demand network", "the theory-of-mind network", "the intuitive physics network", "the early visual cortex".

**Results sentences.** Three paraphrases, identical across candidates apart from the regions, scores averaged over paraphrases: "Relative to fixation, the task most strongly engaged {regions}." / "Task-related responses were largest in {regions}." / "Across both conditions, activity was strongest in {regions}." The result concerns the task as a whole relative to fixation, matching the TRIBE domain localization.

**Scoring.** Primary: perplexity of the full text (prefix + abstract + results sentence), as in BrainBench; per task and style, the candidate with the lowest mean log perplexity over paraphrases is the pick. Robustness: summed log-probability of the results-sentence tokens given the abstract (tokens attributed by character offsets). Preference per candidate = minus (log perplexity − its mean over the five candidates), so 0 = average; for the robustness measure, log-probability minus its mean over candidates.

**Statistics.** Accuracy over the 46 tasks (pick = the task's domain) and balanced accuracy over domains. One-sided p from 10,000 permutations of the task-to-domain labels, which keeps the model's picks and the number of tasks per domain, so a model that favours one candidate for every task is not rewarded. Confusion matrices (domain × pick) and per-domain preference bars (`plots/`).

**No base-model control** (decision: Andrea): only BrainGPT is scored.

**Testing.** Unit tests: `tasks.tsv` covers all 46 tasks with a distinct manipulation per task; each abstract contains the sampled item's correct and incorrect versions; candidates differ only in the region phrase; token attribution in scoring (stub model and tokenizer); analysis on synthetic scores (a perfect scorer gives accuracy 1 and p < .001; a scorer that always picks the distractor gives accuracy 0 and p = 1). End-to-end run of `analyze.py` on fake scores; outputs deleted.

### 2026-10-01 — Redesign of the abstracts (Andrea)

The first version used one neutral, construct-free description of the items' form per task plus two correct example items. Andrea's review: these did not read like abstracts, said nothing about the contrast (grammatical vs ungrammatical, plausible vs implausible), made the physics tasks identical (no mention of temperature, buoyancy, etc.), and avoiding construct names was the wrong goal, since the construct is what the literature prior should respond to. Replaced by the task-specific abstracts described above (`tasks.tsv` replaces `descriptions.tsv`; the banned-word check was removed). The `agent` task description was corrected after inspecting its items: it covers preferences, intentions, effort and beliefs inferred from behaviour and perception, not only beliefs.
