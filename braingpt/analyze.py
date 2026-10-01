"""Summarize BrainGPT scores (CPU, local): picks, confusion matrices, accuracy with permutation tests, and figures.

Primary measure: full-abstract perplexity (BrainBench criterion). Robustness: log-probability of the results
sentence given the abstract. Both for anatomical and network-name candidates.
Writes results/{preferences,picks,summary}.csv and plots/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bglib import CANDIDATES, DOMAINS, HERE, accuracy_permutation, balanced_accuracy, confusion, load_config, picks, \
    task_preferences
from tribeloc.plotting import DOMAIN_COLORS, DOMAIN_LABELS, apply_style, save_fig, style_axes

CAND_COLORS = {**DOMAIN_COLORS, "visual": "#7f7f7f"}
CAND_LABELS = {"Lan": "Language", "MD": "MD", "ToM": "ToM", "phys": "Physics", "visual": "Visual"}
MEASURES = {"ppl": "full-abstract perplexity", "logprob_result": "results-sentence log-probability"}


def plot_confusion(p, style, measure):
    C = confusion(p)
    fig, ax = plt.subplots(figsize=(4.2 * .85, 3.2 * .85))
    ax.imshow(C.to_numpy() / C.sum(1).to_numpy()[:, None], cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(len(DOMAINS)):
        for j in range(len(CANDIDATES)):
            frac = C.iloc[i, j] / C.iloc[i].sum()
            ax.text(j, i, C.iloc[i, j], ha="center", va="center", fontsize=9, color="white" if frac > .5 else "black",
                    weight="bold" if CANDIDATES[j] == DOMAINS[i] else "normal")
    ax.set_xticks(range(len(CANDIDATES)), [CAND_LABELS[c] for c in CANDIDATES], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(DOMAINS)), [f"{DOMAIN_LABELS[d]} tasks" for d in DOMAINS], fontsize=9)
    for t, d in zip(ax.get_yticklabels(), DOMAINS):
        t.set_color(DOMAIN_COLORS[d])
    ax.set_xlabel("BrainGPT's pick", fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save_fig(fig, HERE / "plots" / f"confusion_{style}_{measure}")


def plot_preferences(pref, measure):
    styles = list(pref["style"].unique())
    fig, axes = plt.subplots(len(styles), len(DOMAINS), figsize=(8.8 * .85, 2.9 * .85 * len(styles)), squeeze=False)
    rng = np.random.default_rng(0)
    for r, style in enumerate(styles):
        q = pref[pref["style"] == style]
        for ax, d in zip(axes[r], DOMAINS):
            g = q[q.domain == d]
            for x, c in enumerate(CANDIDATES):
                target = c == d
                v = g[c].to_numpy()
                ax.bar(x, v.mean(), width=.7, color=CAND_COLORS[c], alpha=.6 if target else .3, edgecolor="black",
                       linewidth=1.2 if target else .8, zorder=2)
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color="black", capsize=3, lw=1.1, zorder=4)
                ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=8, color=CAND_COLORS[c], edgecolor="black", lw=.3,
                           alpha=.9 if target else .4, zorder=3)
            ax.axhline(0, color="black", lw=.8, zorder=1)
            ax.set_xticks(range(len(CANDIDATES)), [CAND_LABELS[c] for c in CANDIDATES] if r == len(styles) - 1 else [],
                          rotation=30, ha="right", fontsize=9)
            if r == 0:
                ax.set_title(f"{DOMAIN_LABELS[d]} tasks", fontsize=11, weight="bold", color=DOMAIN_COLORS[d])
            style_axes(ax)
        axes[r, 0].set_ylabel(f"Preference ({style})", fontsize=10)
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, HERE / "plots" / f"preferences_{measure}")


def main():
    apply_style()
    cfg = load_config()
    scores = pd.read_csv(HERE / "results" / "scores.csv")
    rng = np.random.default_rng(cfg["examples"]["seed"])
    summary, all_pref, all_picks = [], [], []
    for measure in MEASURES:
        pref = task_preferences(scores, measure)
        pref.insert(0, "measure", measure)
        all_pref.append(pref)
        plot_preferences(pref, measure)
        for style, q in pref.groupby("style", sort=False):
            p = picks(q)
            all_picks.append(p)
            acc, pval = accuracy_permutation(p, cfg["n_permutations"], rng)
            rec = dict(measure=measure, style=style, accuracy=acc, p_perm=pval, balanced_accuracy=balanced_accuracy(p),
                       n_tasks=len(p), picked_visual=int((p.pick == "visual").sum()))
            for d in DOMAINS:
                rec[f"accuracy_{d}"] = p[p.domain == d].correct.mean()
            summary.append(rec)
            plot_confusion(p, style, measure)
            print(f"\n[{MEASURES[measure]} | {style}] accuracy {acc:.2f} (p = {pval:.4f}), "
                  f"balanced {rec['balanced_accuracy']:.2f}\n{confusion(p).to_string()}", flush=True)
    pd.concat(all_pref).to_csv(HERE / "results" / "preferences.csv", index=False)
    pd.concat(all_picks).to_csv(HERE / "results" / "picks.csv", index=False)
    pd.DataFrame(summary).to_csv(HERE / "results" / "summary.csv", index=False)


if __name__ == "__main__":
    main()
