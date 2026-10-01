# TRIBE localization of the 46 LLM-modularity tasks

This repository tests whether the 46 tasks of the LLM-modularity study (Language, Multiple Demand, Physics, Theory of Mind) engage the brain networks they are meant to target, using TRIBE v2 as a predicted average-subject brain. The question comes from a reviewer: without brain data we cannot know whether these items recruit the target networks. Instead of localizer tasks, we find the cortical locations whose predicted responses discriminate correct (plausible) from incorrect (implausible) completions of each task's items, and then ask whether these locations fall in the expected network parcels.

This README is the running record of every decision, for the methods section. Entries are dated.

## Status

- [x] Old pilot removed (2026-09-30)
- [x] Stimulus table built and tested (2026-09-30): 45,542 items, 182,168 stimuli
- [x] Parcels projected to fsaverage5 (2026-09-30)
- [x] TRIBE benchmark (2026-09-30, job 24506323) and full inference (2026-10-01, array job 24508074; 183/183 shards)
- [x] Shard quality check (2026-10-01)
- [x] Univariate domain contrasts, whole-parcel summaries, spin tests, figures (2026-10-01)
- [x] Multivariate analysis dropped and removed (2026-10-01)
- [ ] Text localizers (language, MD, ToM, physics): stimuli, TRIBE inference, fROI definition
- [ ] fROI-based bar plots

## Repository layout

```
config/analysis.yaml     all analysis parameters (single source of truth)
data/parcels/            original MNI parcel NIfTIs + label tables (see data/parcels/README.md)
data/parcels/fsaverage5/ projected parcels + cortex mask (npz) and projection audit (csv)
data/stimuli/            stimulus table (one row per stimulus) and per-task summary
data/external/           pinned clone of LLM_Modularity (git-ignored, re-created by script)
src/tribeloc/            library: stimuli, parcels, inference, group (statistics), surface (spin-test sphere), plotting
scripts/                 numbered pipeline steps
slurm/                   Engaging environment setup and jobs
tests/                   pytest
results/analysis/        small derived results (tracked); results/tribe/ holds the shards (cluster only, git-ignored)
plots/                   figures (SVG with editable text + PNG)
logs/                    Slurm logs (tracked)
```

## How to run

Locally (conda env `analysis`, with `pip install -e '.[test]'`):

```bash
scripts/00_fetch_source.sh            # LLM_Modularity at the pinned commit
python scripts/01_build_stimuli.py    # data/stimuli/stimuli.csv.gz
python scripts/02_project_parcels.py  # data/parcels/fsaverage5/
python -m pytest
```

On Engaging (GPU), from the project directory `/orcd/data/evelina9/001/USERS/devar_ag/TRIBE-localizer`, after copying the repository there:

```bash
mkdir -p logs
bash slurm/setup_env.sh                     # once, in a compute allocation: creates conda env `tribe`
export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new
hf auth login                               # once, with HF_HOME as in slurm/common.sh; Llama-3.2-3B is gated
hf download meta-llama/Llama-3.2-3B config.json   # check access
sbatch slurm/benchmark.sbatch               # shards 17 (smallest) and 82 (largest)
python scripts/03_run_tribe.py --list       # number of shards (183)
sbatch slurm/inference.sbatch               # 46 array tasks x 4 shards, at most 6 GPUs at a time
```

Then the analysis (CPU), and push the small results back:

```bash
sbatch slurm/qc.sbatch            # shard quality check -> results/analysis/qc/
sbatch slurm/post.sbatch          # 06 univariate maps, 08 parcel summaries + spin tests, 09 figures -> plots/
git add results/analysis plots logs && git commit -m "Analysis results" && git push
```

Steps 08 and 09 need only `results/analysis/univariate.npz` and also run locally.

Each shard is saved as `results/tribe/shards/shard_NNNN.npz` (`stim_id` plus one float32 stimulus × 20,484 array per window). Shards already present are skipped, so failed array tasks can simply be resubmitted.

## Decision log

### 2026-09-30 — Why the earlier localizer pilot was abandoned

A collaborator's pilot (package `andrea_tribe_pilot_20260930`, now deleted) defined fROIs within the parcels from simulated localizer contrasts (sentences vs nonwords, false belief vs false photograph, hard vs easy spatial working memory, TowerLoc physics vs colour) and evaluated the 46 tasks in them. Language and ToM localizers behaved sensibly; MD and Physics did not. Diagnosis:

- TRIBE v2 maps stimulus features (frozen Llama-3.2-3B text, V-JEPA2 video, Wav2Vec-BERT audio) to predicted fMRI on fsaverage5. It has no input for task, instruction or goal. Its training data are passive naturalistic viewing and listening only: CNeuroMod (Friends and movie10 only; the loader in `tribev2/studies/algonauts2025.py` excludes CNeuroMod's task data), BOLD Moments, Lebel2023 podcasts, Wen2017 videos. The authors state: "the model currently treats the brain as a passive observer of naturalistic stimuli; it does not yet model the brain as an active agent producing behavior" (d'Ascoli et al., 2026, arXiv:2605.04326; quote to be re-checked against the PDF before it enters the paper).
- TowerLoc's two conditions share the same videos and differ only in task, so in TRIBE the contrast reduced to the two 1-s cue phrases: a tower-invariant map (98.9% of paired-difference variance), not located in physics parcels.
- The MD spatial working-memory adaptation was a passive video; hard vs easy differs mainly in the number of squares shown, and the contrast peaked in early visual cortex (calcarine, lingual, cuneus), with no enrichment in MD parcels.
- Held-out "validation" of these contrasts was uninformative: TRIBE is deterministic, so any systematic stimulus difference replicates across halves.

Conclusion: task-demand localizers cannot be simulated in a stimulus-only encoding model. We localize with the task items themselves.

### 2026-09-30 — Design of the new analysis

**Stimuli.** Source: `data/{Lan,MD,ToM,phys}/*.json` from github.com/Pengrui-Han/LLM_Modularity, pinned at commit `e3ac7fbb3a6caea05c88343a8de6ec04a4035db8` (2026-07-24). Each item has a clean and a corrupted problem, and two answers that swap roles: in all items of all 46 tasks, `clean_correct == corrupted_incorrect` and `clean_incorrect == corrupted_correct`. Each item therefore yields a fully crossed 2×2 of four stimuli (problem version × answer). Two are correct (clean problem + clean answer, corrupted problem + corrupted answer) and two incorrect. Correctness is the interaction: every problem string and every answer string occurs once as correct and once as incorrect within an item, so word identity, problem length and answer frequency cannot predict the label.

**Text presented to TRIBE.** Raw problem + answer, without the task prompt prefixes or suffixes used for the LLMs (decision: Andrea). Language tasks: answers carry their own leading space and are appended directly; wug answers are suffixes that fuse with the last word (`utetorn` + `ities`), so the answer onset is the onset of that fused word. All other tasks: answers were a separate chat turn for the LLMs, so they are joined to the problem with one space. Text only, no synthesized speech (decision: Andrea). Words are whitespace tokens shown at 0.5 s each, as in the pilot.

**Items.** All items of all 46 tasks are used (decision: Andrea), after a uniqueness filter (decision: Andrea). The release contains three kinds of repetition, none created by our string joining (whitespace normalisation merges no items; shared strings are the same whether compared as raw (problem, answer) pairs or as joined text): 673 exact duplicate rows (e.g. subject_verb_agreement 260, number_sequence 168, code_B 135, npi 79); 1,733 mirror pairs with clean and corrupted swapped, which are distinct patching directions for the LLM analysis but contain the same four sequences (wug is entirely mirrored, 1,332 pairs); and 1,890 stimulus strings reused by different items (e.g. agent 458, hypernymy 392; hypernymy reuses corrupted problems so heavily that its 577 items form only 6 linked groups). TRIBE is deterministic, so a repeated string in training and test data would leak. Rule: going through tasks in config order and items in source order, an item is kept only if its four texts are distinct and none appeared in any earlier kept item, in any task. The filter is dataset-wide because 11 code_B items reuse code_list stimuli verbatim. Result: 49,430 → 45,542 items, 182,168 stimuli, every text unique. Largest losses: wug 2,664 → 1,329 and npi 304 → 152 (mirror duplicates), hypernymy 577 → 214, number_sequence 946 → 488, code_B 1,000 → 650, subject_verb_agreement 1,100 → 759. Per-task counts: `data/stimuli/task_summary.csv`. Smallest task: npi, 152 items (this sets the matched N for classification).

**Mean words per stimulus by domain** (descriptive; no length covariate): Lan 5.9, MD 21.6, phys 47.7, ToM 36.8.

**Halves** (used only by the abandoned multivariate analysis; the column is kept so the inference input is unchanged). Within each task, items are randomly split in two halves (seed 20260930 with the task index; first ⌊n/2⌋ of a permutation → half 1). All four stimuli of an item are in the same half.

**Analysis windows** (decision: Andrea; fixed before inference because only window means are stored). `answer`: answer onset → +6 s (multivariate, main). `full`: stimulus onset → stimulus end (univariate, the pilot's convention). `full_tail`: stimulus onset → stimulus end + 4 s (sensitivity). TRIBE's output is at 1 Hz and already shifted 5 s to compensate for haemodynamic lag (checkpoint config `neuro.offset: 5.0`); no further lag is added. A window includes output samples t with lo ≤ t < hi.

**TRIBE inference.** TRIBE v2 (github.com/facebookresearch/tribev2 at commit `af58661791a351a448a489042a28f6c37e1c14b7`; checkpoint `facebook/tribev2` from Hugging Face). Text pathway only (`features_to_use: [text]`): word features from Llama-3.2-3B, contextualized, with each word's context being the stimulus text up to and including that word. Each stimulus is an isolated 100-s timeline with onset at 10 s, as in the pilot; TRIBE cuts one 100-TR segment per timeline (checkpoint `duration_trs: 100`), so shortening timelines would not save compute. Words are presented every 0.5 s (longest stimulus: number_sorting, 77 words, ending at 48.5 s). No zero-feature baseline is computed: decoding and between-domain contrasts do not need it. Inference runs in 183 shards of about 1,000 stimuli, each holding whole items, with a fresh node-local feature cache deleted after each shard (TRIBE caches 20 Llama layers per word, about 245 KB per word; total 5.16M words).

**Multivariate measure (abandoned 2026-10-01; code removed).** The original main analysis was a whole-cortex surface searchlight (10-mm geodesic discs) measuring the crossnobis distance between correct and incorrect completions of each item's 2×2 (correct-minus-incorrect difference, Ledoit–Wolf whitening, sign-flip permutation null with max-statistic FWE), plus matched-N shrinkage-LDA classification. It was validated on synthetic data with a planted effect. On the real predictions it did not localize (see Results, Round 1), and Andrea decided not to report it. The code was removed from the repository after commit `a4f9b56`, where it can still be found (scripts 04, 05, 07; `src/tribeloc/rsa.py`).




**Univariate measure.** At every vertex, a domain's mean response minus the mean of the other three domains, in the `full` window (`full_tail` as sensitivity). Task maps average all stimuli of a task (correct and incorrect); domain means weight tasks equally. Inference: the 46 task-to-domain labels are permuted 10,000 times; one-sided maximum-statistic FWE over cortical vertices. TRIBE outputs are compared between domains only, so no zero-input baseline is needed. No length covariate (decision: Andrea); mean word count per domain is reported descriptively.

**Benchmark** (2026-09-30, Engaging job 24506323, one A100 80GB). Smallest shard (17, subject_verb_agreement; 1,000 stimuli, 3,068 words): 90 s. Largest (82, number_sorting; 75,056 words): 856 s, of which about 13 min was Llama feature extraction over 36,440 unique (word, context) pairs, since the two answers to a problem share the problem's words. Estimated total about 20 GPU-hours; output about 0.24 GB per shard (43 GB in all). The neuralset warning "LabelEncoder has only found one label" is expected and harmless: `TribeModel.from_pretrained` sets `average_subjects=True`, in which the output layer uses the shared average-subject weights and ignores the subject label (neuraltrain 0.0.2, `SubjectLayersModel.forward`), and the checkpoint has `subject_embedding: false`.

**Full inference** (2026-10-01, array job 24508074, at most 6 A100s at a time). All 183 shards saved (181 in this job, 2 from the benchmark); no errors in any log, only the expected neuralset subject-label warnings. Most shards took 20–260 s; the number_sorting shards took 9–15 min because bracketed number lists tokenize into many tokens per word.

**Parcels on the surface.** Original MNI NIfTIs (checksums verified) projected to fsaverage5 with neuromaps 0.0.7 registration fusion (`mni152_to_fsaverage`, `fsavg_density='10k'`, nearest neighbour). Vertex counts match the pilot's projection exactly. Every parcel projects onto cortex. Nine vertices of PHYSICS lSPL fell in the right hemisphere (midline voxels) and are removed so each parcel stays in its own hemisphere. Cortex mask: Destrieux fsaverage5 labels excluding Unknown and Medial_wall (18,715 of 20,484 vertices). Audit: `data/parcels/fsaverage5/projection_audit.csv`.

**Parcel summaries.** All parcels kept; probability atlases (LanA etc.) dropped. Physics was first analysed with both parcel sets, `PHYSICS` and `PHYSICS_Kean`; from 2026-10-01 only `PHYSICS` is used (decision: Andrea). Peak decodability per parcel: items are split into two random halves per task (seeded); the top 10% of searchlight centres within each parcel (rounded up) are selected on one half's crossnobis map and their mean crossnobis distance is evaluated on the other half's map, then the halves are swapped and averaged. This is done per task and parcel; network values average parcels equally, domain values average tasks equally, and the SEM is across tasks. The whole-parcel mean (no selection, all items) is reported alongside. Item split chosen over a clean/corrupted split because each half keeps the full balanced 2×2 and the halves share no strings (decision: Andrea). Whole-brain maps for display use all data.

**Correspondence with the parcels** (decision: Andrea). Overlap between each domain's whole-cortex map and that network's parcels is tested against a spin-test null (Alexander-Bloch et al., 2018, NeuroImage 178:540–551): 1,000 random rotations of the fsaverage5 sphere, the right hemisphere rotated by the mirror image of the left rotation; each vertex takes the value of the vertex nearest its inverse-rotated position, and medial-wall values rotated into cortex are dropped. Statistic: the fraction of the top 10% of cortical vertices of a domain's crossnobis map that fall in a network's parcels (enrichment = this fraction divided by the network's share of cortex); one-sided p = (1 + #rotations with overlap ≥ observed) / 1,001. All five parcel sets are tested against all four domain maps.

**Figures.** Lateral and medial surface maps per domain for the multivariate and univariate results with parcel outlines; domain × network matrix of held-out peak decodability; per-parcel plots.

## Results

### 2026-10-01 — Round 1

**Shard quality check** (`results/analysis/qc/`). All 183 shards complete (182,168 stimuli), all values finite, predictions vary across stimuli in every shard, and no two stimuli share a prediction. The item-level correct-minus-incorrect difference in the `answer` window is small relative to the spread across stimuli: mean |d| / SD across stimuli 0.01–0.33 per task, largest for Language tasks (e.g. hypernymy 0.33, subject–verb agreement 0.23), where the single-word answer is a large part of a short stimulus.

**Multivariate correctness maps are not localizing.** Every cortical vertex is FWE-significant for every domain (18,715/18,715). The four domain crossnobis maps are nearly identical (pairwise r = 0.987–0.995 across vertices). All four peak in the same ventral and orbital regions (gyrus rectus, orbital gyri, right parahippocampal, lingual and inferior temporal gyri, plus left lateral superior temporal gyrus); these are regions of strong susceptibility dropout in human fMRI, where TRIBE's training signal is weakest. Classification accuracy is nearly uniform across cortex (mean / max: Language 0.754 / 0.792, MD 0.587 / 0.615, Physics 0.548 / 0.564, ToM 0.612 / 0.641). Held-out parcel fROIs give the same network profile for every domain (ToM ≥ Language > MD > Physics parcels), with no domain × network interaction. Spin tests: the top 10% of every domain map is under-represented in every parcel set (enrichment 0.02–0.74; all p_spin > 0.6). Interpretation: TRIBE's vertex outputs are read out from one shared latent state, so correctness information encoded there reaches every vertex. Whitened (noise-normalized) decodability then reflects the readout geometry rather than where a computation takes place.

**Univariate domain contrasts localize for MD, Physics and ToM, not for Language.** Enrichment of the top 10% of cortex (`full` window) in the target parcels: MD > others in MD parcels 3.13; Physics > others in PHYSICS 3.91 and PHYSICS_Kean 2.89 (also MD 1.96, which overlaps physics); ToM > others in TOM 3.43 (and Language parcels 4.96); Language > others in Language parcels 0.02. Peaks: MD in the intraparietal sulcus, angular gyrus and middle frontal gyrus; Physics in the supramarginal gyrus, lateral occipital cortex and pre/postcentral sulci; ToM in bilateral STS/STG and superior frontal gyrus; Language in medial cortex (pericallosal sulcus, precuneus, anterior cingulate). FWE-significant vertices: Language 142, MD 439, Physics 577, ToM 495. `full_tail` gives the same pattern. Spin tests for the univariate maps are not computed yet.

**The Language univariate result is confounded with stimulus length.** Across the 46 tasks, the mean predicted response in the Language parcels correlates with mean stimulus length (Spearman ρ = 0.69, p = 1e-7; within Language tasks ρ = 0.83, within MD ρ = 0.92). Language-parcel response by domain follows length: Lan 0.173 (5.9 words), MD 0.205 (21.6), Physics 0.234 (47.7), ToM 0.272 (36.8). The response averaged over a short stimulus starting from silence is lower, so domain contrasts for Language cannot be separated from length with these stimuli. (The earlier decision not to use a length covariate is revisited after these results.)

### 2026-10-01 — Decisions after round 1 (Andrea)

- **The univariate domain contrast is the main analysis.** The multivariate correctness analysis is not reported in this paper: its maps are not localizing in TRIBE (see Round 1), which is a property of the model that may be studied separately. Code and round-1 outputs are kept (`04_searchlight.py`, `05_domain_maps.py`, `07_parcel_summary.py`, `results/analysis/searchlight/`, `domain_maps.npz`, `parcel_heldout.csv`, `network_domain_heldout.csv`, `spin_tests.csv`).
- **Language is reported with the length confound stated**, no adjustment (option b): every task is text, the language network responds to all of them, and its predicted response grows with the amount of text. Stimulus lengths do not overlap between Language (3–11 words) and Physics (≥15 words), so no adjustment can separate domain from length.
- **Spin tests for the univariate maps** added.

**Univariate parcel analysis** (`08_univariate_summary.py`). For each parcel set (target domain: Language parcels ↔ Language tasks, MD ↔ Formal, ToM ↔ Social, both physics sets ↔ Physics) and window: each task's response is averaged over a parcel's cortical vertices, then over parcels (equal weight). The target domain's mean over tasks is compared with the mean of the other three domain means, and with each other domain separately; one-sided p-values come from 10,000 permutations of the task-to-domain labels (tasks are the unit, n = 46). Held-out fROI version (when item-half maps are available): within each parcel, the top 10% of vertices by the target domain's contrast (target minus the mean of the other domain means) in one item half; every task's response is measured there in the other half; then halves are swapped and averaged. Spin tests as described above, on each domain's contrast map against each parcel set.

**Figures** (`09_figures.py`, `plots/`; domain names and colours as in the LLM-modularity paper: Language, Formal = MD tasks, Physics, Social = ToM tasks). `univariate_maps_<window>`: each domain minus the other three, inflated surface, lateral and medial views of both hemispheres, target parcels outlined (physics: the `PHYSICS` set), symmetric colour scale at each row's 99th percentile of |value|, medial wall grey. `network_bars_<selection>_<window>`: response of each parcel set to the four domains (bars = mean over tasks ± SEM; dots = tasks), with brackets for the target domain against each other domain (one-sided permutation p, uncorrected: * < .05, ** < .01, *** < .001). `enrichment_<window>`: overlap of each domain map's top 10% with each parcel set, relative to the parcel set's share of cortex (* spin p < .05). `length_language_<window>`: language-parcel response against mean words per stimulus across the 46 tasks.

### 2026-10-01 — Univariate results (whole parcels)

`full` window; target domain minus the mean of the other three (permutation p, n = 46 tasks): MD parcels, Formal +0.042 (p = .0008); ToM parcels, Social +0.047 (p = .0002); Physics parcels, Physics +0.087 (p = .0001); Physics (Kean) parcels, Physics +0.069 (p = .0001); Language parcels, Language −0.056 (p = .998). Pairwise: MD parcels prefer Formal over Social (p < .001) but not significantly over Language or Physics tasks; ToM parcels prefer Social over all three (Language p < .05, Formal and Physics p < .001); both physics sets prefer Physics over all three (p < .001). Spin tests (top 10%): Formal map in MD parcels 3.13× (p = .001); Physics map in PHYSICS 3.90× (p = .001), PHYSICS_Kean 2.90× (p = .013), and MD 1.96× (p = .013; physics and MD parcels overlap); Social map in ToM 3.43× (p = .009) and Language 4.96× (p = .001); Language map in Language parcels 0.02× (p = 1.0). `full_tail` gives the same conclusions. In the length scatter, at matched lengths Language tasks evoke a larger language-parcel response than Formal tasks (e.g. about 0.18 vs 0.11 at 5 words), but this is descriptive.

### 2026-10-01 — Transposed view: which parcels prefer each domain (Andrea's request)

`domain_bars_selectivity_<selection>_<window>` (main) and `domain_bars_response_<selection>_<window>` (raw). One panel per task domain; bars = the five parcel sets. Raw responses are not comparable across parcel sets, because the language parcels respond about 4× more than the others to every domain (about 0.2 vs 0.05). The main version therefore plots selectivity: for each task, its response in a parcel set minus that set's mean response to the other three domains (domains weighted equally, tasks equally within domain). This is the per-task version of the network contrast above (averaging over a domain's tasks gives the same number), and a constant offset of a parcel set cancels. Test: target parcel set against each other set, one-sided paired sign-flip across the domain's tasks (exact for Language, Physics and Social, 8–9 tasks, minimum p = 1/256 or 1/512; 10,000 random flips for Formal, 20 tasks); uncorrected. Physics: PHYSICS is the target set, with PHYSICS_Kean shown but not tested against it. Stats: `results/analysis/univariate/domain_stats.csv`; per-task selectivity: `task_selectivity_<selection>_<window>.csv`.

Results (`full`, whole parcels; selectivity, target set first): Formal tasks: MD +0.042 > Language +0.001 (p = .022), ToM −0.035, Physics −0.028, Physics (Kean) −0.011 (all p < .001). Physics tasks: Physics +0.087, Kean +0.069 > Language +0.004, MD +0.003, ToM −0.028 (all p = .002, the exact minimum). Social tasks: ToM +0.047 > MD, Physics, Kean (all p = .002), but not > Language parcels (+0.052, p = .81). Language tasks: Language parcels −0.056, the least selective set (length confound). `full_tail`: the same, except MD vs Language parcels for Formal tasks becomes p = .065.

### 2026-10-01 — Clean-up and figure conventions (Andrea)

- Multivariate analysis removed from the repository (see the note in the design section).
- `PHYSICS_Kean` dropped for good: one physics parcel set (`PHYSICS`) only. The NIfTI stays in `data/parcels/` as part of the original parcel bundle, but it is no longer projected or analysed. Projections of the other sets are unchanged.
- Figures: no figure titles (panel labels only); bar plots 40% taller, with one shared y-axis per figure; axes and colour bars labelled in TRIBE's units. **Units:** TRIBE was trained on BOLD that was detrended and z-scored per vertex and run (checkpoint config `neuro.cleaning: standardize: zscore_sample, detrend: true`). Predictions are therefore in z units, i.e. standard deviations of the vertex's training signal, not % signal change, and 0 is the vertex's mean during naturalistic stimulation, not fixation. Labels: "Predicted BOLD (z)"; maps "Δ predicted BOLD (z)".
- The held-out item-half fROI analysis (planned but never run) is dropped. fROIs will be defined with independent text localizers instead.
