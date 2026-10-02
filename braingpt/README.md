# BrainGPT literature prior for the 46 tasks

A separate, complementary analysis to the TRIBE localization in the parent repository. TRIBE asks where text drives predicted brain responses. Here we ask which brain network the neuroscience literature would expect a study of each task to report, using BrainGPT (a LoRA adapter for Mistral-7B-v0.1 trained on PubMed Central neuroscience papers, 2002–2022; Luo et al., Nature Human Behaviour; BrainBench). BrainGPT is a scorer, not a predictor of activation: for an abstract with several candidate results, the candidate with the lowest perplexity is the one the model finds most plausible, and the perplexity gap is its confidence. This is evidence about the literature's expectations (a stand-in for expert consensus), not a measurement.

## Layout

```
config.yaml           model, prompt prefix, abstract template, result paraphrases, candidate regions
tasks.tsv             hand-written per task: background (what it studies), manipulation (what makes an item correct or not), labels
bglib.py              abstract construction, scoring, analysis helpers
build_abstracts.py    -> data/abstracts.csv (all texts to score), data/abstracts_preview.txt (one abstract per task)
score.py              -> results/scores.csv (GPU)
analyze.py            -> results/{preferences,picks,summary}.csv, plots/ (confusion matrices, assignment counts)
plot_assignments.py   -> plots/assignment_diagram_<measure>_<style> (main figure)
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

### 2026-10-01 — First results and calibration (Engaging job 24596498)

**Uncalibrated results.** Scoring took 105 s for 1,380 texts. Accuracy over the 46 tasks was above chance under the label-permutation test but low: full-abstract perplexity 0.43 (anatomical, p = .003) and 0.39 (network names, p < .001); results-sentence log-probability 0.37 and 0.26. The picks were dominated by the candidates rather than the tasks: with anatomical phrasing and perplexity, ToM was picked for 30 of 46 tasks and Language and Physics never; with network names the distractor "the early visual cortex" was picked for 21 tasks; the results-sentence log-probability picked Language for 25 tasks. Candidate identity explained 77% (anatomical) and 85% (network names) of the variance of the results-sentence log-probability across tasks and candidates, after removing each task's mean. Mean token counts of the results sentences differ between candidates (anatomical 23–34, network names 12–16), and some phrasings are much rarer than others in any context ("the intuitive physics network" is 9.6 log units below the mean of the five candidates on average). BrainBench compares minimal-pair abstracts of similar length, where these differences do not arise; our candidates are different strings, so their surface probability competes with the task information (surface form competition).

**Calibration (decision: Andrea).** Domain-conditional PMI (Holtzman, West, Shwartz, Choi & Zettlemoyer, 2021, EMNLP, "Surface Form Competition: Why the Highest Probability Answer Isn't Always Right"): each results sentence is also scored after a neutral abstract with the same template but no task information ("We used fMRI to measure brain responses while healthy adult participants read short written items, presented one word at a time, and judged whether each item was correct. Items were presented in blocks, and responses were compared with a fixation baseline."), and the calibrated score is log p(result | task abstract) − log p(result | neutral abstract). The same tokens enter both terms, so phrase length and baseline frequency cancel; the score measures how much the task's abstract raises the probability of each region set. This is now the primary measure; full-abstract perplexity (the BrainBench criterion) and the uncalibrated log-probability are still reported. 30 additional texts (neutral abstract × 2 styles × 5 candidates × 3 paraphrases). Test: a constant advantage for one candidate in every abstract wins the uncalibrated measure for every task but cancels in the PMI.

### 2026-10-01 — Calibrated results (PMI)

Accuracy over the 46 tasks (chance assessed by permuting task-to-domain labels, 10,000 permutations):

| Candidates | Measure | Accuracy | Balanced | p | Language | Formal | Physics | Social |
|---|---|---|---|---|---|---|---|---|
| anatomical | PMI (primary) | 0.76 (35/46) | 0.77 | < .001 | 4/8 | 14/20 | 8/9 | 9/9 |
| network names | PMI | 0.63 (29/46) | 0.66 | < .001 | 1/8 | 10/20 | 9/9 | 9/9 |
| anatomical | full-abstract perplexity | 0.43 | 0.39 | .002 | 0/8 | 11/20 | 0/9 | 9/9 |
| network names | full-abstract perplexity | 0.39 | 0.36 | < .001 | 0/8 | 9/20 | 0/9 | 9/9 |

Calibration removes the candidate biases: with anatomical candidates the distractor is picked once (physics_brightness, which concerns light), against 12–21 times without calibration. Errors are structured. Anatomical: the six code tasks and number_sorting go to the physics candidate (dorsal premotor cortex, supplementary motor area, superior parietal lobule), which overlaps anatomically with the dorsal fronto-parietal part of the MD system; four of the eight language tasks go to MD (Language and MD preferences are close for the agreement tasks). Network names: all logic, equation and code tasks go to "the intuitive physics network", and seven of the eight language tasks to "the theory-of-mind network". The anatomical result is the more informative one (network names are mostly label matching, see the design notes), and it is the main result.

### 2026-10-01 — Why language tasks went to ToM under PMI; across-task contrast (decision: Andrea)

**Diagnosis.** The neutral abstract was not neutral about language: it describes participants reading written items and judging them, which by itself makes BrainGPT expect the language network (log-probability of the language results sentence after the neutral abstract, relative to the mean of the five candidates: +6.3 with network names, +6.5 anatomical). Uncalibrated, BrainGPT picked the language candidate for 8/8 language tasks under both phrasings, but the language abstracts raised the language sentence only a little above that ceiling (+0.6), less than they raised the theory-of-mind sentence from a low starting point (+1.5; the agreement items contain names and people). PMI therefore flipped most language tasks to ToM. The calibration context must control surface form only, and the neutral abstract also carried content.

**Across-task contrast (now primary).** For each task and candidate, the results-sentence log-probability (mean over paraphrases) minus the mean over the other three domains' tasks for the same candidate (domain means weight tasks equally), i.e. the same "domain minus the other three" logic as the TRIBE contrasts. All abstracts share the template, so the reading-task component, phrase length and phrase frequency cancel without a hand-written neutral text. The pick is the candidate with the highest contrast. Because the reference depends on the domain labels, the permutation test recomputes the contrast and the picks under every permutation of the labels (10,000). PMI against the neutral abstract, BrainBench perplexity and the uncalibrated log-probability are still reported. Tests: a bias shared by all abstracts cancels; the contrast matches a hand computation; the permutation test gives p < .001 for a perfect scorer and p > .05 for noise.

**Figures.** The per-domain bar plots now show counts: for each task domain, how many tasks were assigned to each candidate (`plots/assignments_<measure>`; decision: Andrea), next to the confusion matrices.

**Results (across-task contrast).**

| Candidates | Accuracy | Balanced | p | Language | Formal | Physics | Social |
|---|---|---|---|---|---|---|---|
| anatomical | 0.78 (36/46) | 0.81 | < .001 | 8/8 | 14/20 | 5/9 | 9/9 |
| network names | 0.89 (41/46) | 0.94 | < .001 | 8/8 | 15/20 | 9/9 | 9/9 |

Remaining errors: the five code tasks go to the physics candidate under both phrasings (anatomically, the physics regions — dorsal premotor, SMA, superior parietal — overlap the dorsal fronto-parietal MD system); with anatomical candidates, four physics tasks (buoyancy, brightness, solubility, temperature: descriptions of substances, liquids and light) go to early visual cortex, and logic_propositional_1 to the language candidate.

### 2026-10-01 — Assignment diagram (main figure; Andrea)

`plot_assignments.py` → `plots/assignment_diagram_<measure>_<style>.{svg,png}`. The 46 tasks along the bottom (short names and within-domain order as in the LLM-modularity paper), grouped Language, Formal, Physics, Social, with per-domain counts of correct assignments; the five candidate networks along the top, each above its domain's tasks (distractor at the right). Overall accuracy and node counts are not printed in the figure (decision: Andrea); they are in `results/summary.csv`. One curve per task to the network BrainGPT assigned it to, coloured by the task's target network: a curve whose colour differs from the node it reaches is a misassignment (drawn thicker and on top). Main version: `assignment_diagram_contrast_anatomical`.
