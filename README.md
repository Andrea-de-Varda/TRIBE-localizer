# TRIBE localization of the 46 LLM-modularity tasks

This repository tests whether the 46 tasks of the LLM-modularity study (Language, Multiple Demand, Physics, Theory of Mind) engage the brain networks they are meant to target, using TRIBE v2 as a predicted average-subject brain. The question comes from a reviewer: without brain data we cannot know whether these items recruit the target networks. Instead of localizer tasks, we find the cortical locations whose predicted responses discriminate correct (plausible) from incorrect (implausible) completions of each task's items, and then ask whether these locations fall in the expected network parcels.

This README is the running record of every decision, for the methods section. Entries are dated.

## Status

- [x] Old pilot removed (2026-09-30)
- [x] Stimulus table built and tested (2026-09-30): 45,542 items, 182,168 stimuli
- [x] Parcels projected to fsaverage5 (2026-09-30)
- [x] Inference code and Slurm scripts written, helpers tested (2026-09-30)
- [ ] TRIBE throughput benchmark on Engaging
- [ ] Full TRIBE inference
- [x] Searchlight, univariate, parcel-summary and spin-test code written and tested on synthetic data (2026-09-30)
- [ ] Searchlight crossnobis and classification (cluster)
- [ ] Univariate contrasts (cluster)
- [ ] Parcel summaries, spin tests (cluster or local)
- [ ] Figures

## Repository layout

```
config/analysis.yaml     all analysis parameters (single source of truth)
data/parcels/            original MNI parcel NIfTIs + label tables (see data/parcels/README.md)
data/parcels/fsaverage5/ projected parcels + cortex mask (npz) and projection audit (csv)
data/stimuli/            stimulus table (one row per stimulus) and per-task summary
data/external/           pinned clone of LLM_Modularity (git-ignored, re-created by script)
src/tribeloc/            library: stimuli.py, parcels.py, inference.py, data.py, surface.py, rsa.py, group.py
scripts/                 numbered pipeline steps
slurm/                   Engaging environment setup, benchmark and array jobs
tests/                   pytest
results/                 outputs (git-ignored)
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
huggingface-cli login                       # once; Llama-3.2-3B is gated
sbatch slurm/benchmark.sbatch               # shards 17 (smallest) and 82 (largest)
python scripts/03_run_tribe.py --list       # number of shards (183)
sbatch --array=0-45%8 --export=ALL,PER_TASK=4 slurm/inference.sbatch   # size from the benchmark
```

Then the analysis (CPU), and push the small results back:

```bash
sbatch --array=0-45 slurm/searchlight.sbatch                      # one job per task
sbatch --dependency=afterok:<array job id> slurm/post.sbatch      # domain maps, univariate, parcels, spin tests
git add results/analysis && git commit -m "Add analysis results" && git push
```

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

**Halves.** Within each task, items are randomly split in two halves (seed 20260930 with the task index; first ⌊n/2⌋ of a permutation → half 1). All four stimuli of an item are in the same half.

**Analysis windows** (decision: Andrea; fixed before inference because only window means are stored). `answer`: answer onset → +6 s (multivariate, main). `full`: stimulus onset → stimulus end (univariate, the pilot's convention). `full_tail`: stimulus onset → stimulus end + 4 s (sensitivity). TRIBE's output is at 1 Hz and already shifted 5 s to compensate for haemodynamic lag (checkpoint config `neuro.offset: 5.0`); no further lag is added. A window includes output samples t with lo ≤ t < hi.

**TRIBE inference.** TRIBE v2 (github.com/facebookresearch/tribev2 at commit `af58661791a351a448a489042a28f6c37e1c14b7`; checkpoint `facebook/tribev2` from Hugging Face). Text pathway only (`features_to_use: [text]`): word features from Llama-3.2-3B, contextualized, with each word's context being the stimulus text up to and including that word. Each stimulus is an isolated 100-s timeline with onset at 10 s, as in the pilot; TRIBE cuts one 100-TR segment per timeline (checkpoint `duration_trs: 100`), so shortening timelines would not save compute. Words are presented every 0.5 s (longest stimulus: number_sorting, 77 words, ending at 48.5 s). No zero-feature baseline is computed: decoding and between-domain contrasts do not need it. Inference runs in 183 shards of about 1,000 stimuli, each holding whole items, with a fresh node-local feature cache deleted after each shard (TRIBE caches 20 Llama layers per word, about 245 KB per word; total 5.16M words).

**Multivariate measure (main).** Whole-cortex surface searchlight; at every cortical vertex, discriminate correct from incorrect stimuli of a task. Primary measure: cross-validated Mahalanobis (crossnobis) distance between correct and incorrect patterns, using all items. Secondary: cross-validated classification accuracy with the number of items matched across tasks (repeated random subsamples). Cross-validation folds keep all four stimuli of an item together. Per task, then averaged within domain; leave-one-task-out cross-task decoding within domain as a secondary analysis. Parcels play no role in computing the maps, so their correspondence with the parcels can be tested afterwards (decision: Andrea).

**Searchlight implementation** (2026-09-30). Searchlights: for every cortical vertex, the cortical vertices of the same hemisphere within 10 mm, with distance measured along the edges of the fsaverage5 midthickness mesh (mean of white and pial coordinates; edge paths slightly overestimate true geodesic distance). Median 38 vertices per searchlight (range 11–86). Window: `answer`. For each item, d = (clean+A + corrupted+B − clean+B − corrupted+A)/2, the correct-minus-incorrect difference, which cancels item, problem and answer identity. Within a searchlight, the d vectors are whitened by the Ledoit–Wolf-shrunk uncentred second moment of d (Ledoit & Wolf, 2004, J. Multivariate Anal. 88:365–411), and the crossnobis distance is the mean inner product of whitened d over all pairs of distinct items, divided by the number of vertices. This is the leave-one-item-out cross-validated Mahalanobis distance (Walther et al., 2016, NeuroImage 137:188–200), with expectation 0 when correct and incorrect stimuli do not differ; the searchlight approach follows Kriegeskorte, Goebel & Bandettini (2006, PNAS 103:3863–3868). Because "noise" in a deterministic model is item-to-item variability, the effect is whitened by its own item-level variability; the uncentred moment makes the whitening identical under the permutation null. Null: swapping the correct/incorrect labels within an item flips the sign of its d exactly, so 1,000 random per-item sign-flip patterns (shared across vertices within a task) give a permutation null; family-wise error is controlled with the maximum statistic over cortical vertices (Nichols & Holmes, 2002, Hum. Brain Mapp. 15:1–25). Domain maps are the mean of task maps; their null averages task null maps permutation by permutation (flips independent across tasks). Each task is also analysed separately in its two item halves for the parcel analysis.

**Validation of the searchlight code** (2026-09-30, synthetic data). Fake TRIBE predictions for npi (152 items; Gaussian noise, with a correctness effect planted in 200 vertices) were run through `04_searchlight.py`. All 139 FWE-significant searchlights contained planted vertices; none of the searchlights without planted vertices was significant. Crossnobis correlated 0.75 with the fraction of planted vertices in the searchlight. In searchlights without signal, accuracy averaged 0.499 and crossnobis −0.0001, about 0.04 null SDs below zero: estimating the whitening from the same items biases the distance slightly downwards, which is conservative, and the permutation test stays exact because every permutation uses the same whitening. Runtime for npi on one core: 12 s crossnobis, 8 min classification (about 175 µs per LDA fit, the same for every task because N is matched); peak memory 2 GB. The domain, parcel and spin-test scripts were run on per-task copies of this output; the spin null's mean enrichment was ≈1, as expected. All synthetic outputs were deleted.

**Classification** (secondary). Shrinkage LDA (Ledoit–Wolf within-class covariance) on item-centred stimulus patterns (each item's four patterns minus their mean, which uses no label information), classifying correct vs incorrect stimuli; no bias term, since after centring the class means are ±d̄/2. 152 items per task (matched to the smallest task, npi), 20 random subsamples, 5 folds over items. Reported as the mean accuracy over subsamples.

**Univariate measure.** At every vertex, a domain's mean response minus the mean of the other three domains, in the `full` window (`full_tail` as sensitivity). Task maps average all stimuli of a task (correct and incorrect); domain means weight tasks equally. Inference: the 46 task-to-domain labels are permuted 10,000 times; one-sided maximum-statistic FWE over cortical vertices. TRIBE outputs are compared between domains only, so no zero-input baseline is needed. No length covariate (decision: Andrea); mean word count per domain is reported descriptively.

**Parcels on the surface.** Original MNI NIfTIs (checksums verified) projected to fsaverage5 with neuromaps 0.0.7 registration fusion (`mni152_to_fsaverage`, `fsavg_density='10k'`, nearest neighbour). Vertex counts match the pilot's projection exactly. Every parcel projects onto cortex. Nine vertices of PHYSICS lSPL fell in the right hemisphere (midline voxels) and are removed so each parcel stays in its own hemisphere. Cortex mask: Destrieux fsaverage5 labels excluding Unknown and Medial_wall (18,715 of 20,484 vertices). Audit: `data/parcels/fsaverage5/projection_audit.csv`.

**Parcel summaries.** All parcels kept; probability atlases (LanA etc.) dropped. Physics is analysed with both parcel sets, `PHYSICS_Kean` and `PHYSICS`. Peak decodability per parcel: items are split into two random halves per task (seeded); the top 10% of searchlight centres within each parcel (rounded up) are selected on one half's crossnobis map and their mean crossnobis distance is evaluated on the other half's map, then the halves are swapped and averaged. This is done per task and parcel; network values average parcels equally, domain values average tasks equally, and the SEM is across tasks. The whole-parcel mean (no selection, all items) is reported alongside. Item split chosen over a clean/corrupted split because each half keeps the full balanced 2×2 and the halves share no strings (decision: Andrea). Whole-brain maps for display use all data.

**Correspondence with the parcels** (decision: Andrea). Overlap between each domain's whole-cortex map and that network's parcels is tested against a spin-test null (Alexander-Bloch et al., 2018, NeuroImage 178:540–551): 1,000 random rotations of the fsaverage5 sphere, the right hemisphere rotated by the mirror image of the left rotation; each vertex takes the value of the vertex nearest its inverse-rotated position, and medial-wall values rotated into cortex are dropped. Statistic: the fraction of the top 10% of cortical vertices of a domain's crossnobis map that fall in a network's parcels (enrichment = this fraction divided by the network's share of cortex); one-sided p = (1 + #rotations with overlap ≥ observed) / 1,001. All five parcel sets are tested against all four domain maps.

**Figures.** Lateral and medial surface maps per domain for the multivariate and univariate results with parcel outlines; domain × network matrix of held-out peak decodability; per-parcel plots.
