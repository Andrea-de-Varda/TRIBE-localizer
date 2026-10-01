"""Figures for the univariate analysis -> plots/ (SVG with editable text + PNG).

  univariate_maps       each domain minus the other three, on the inflated surface, with the target parcels outlined
  network_bars_<sel>    predicted response of each parcel set to the four task domains (whole parcel; held-out fROI)
  domain_bars_<sel>     transposed: for each task domain, the selectivity of each parcel set for it (and the raw response)
  enrichment            overlap of each domain map's top 10% with each parcel set, with spin-test significance
  length_language       language-parcel response vs stimulus length across the 46 tasks (the Language confound)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

from tribeloc import ROOT, load_config
from tribeloc.parcels import load as load_parcels
from tribeloc.plotting import (DOMAIN_COLORS, DOMAIN_LABELS, DOMAINS, NETWORK_LABELS, NETWORK_TARGET, apply_style,
                               bracket, save_fig, stars, style_axes)

OUT = ROOT / "plots"
MAP_NETWORKS = {"Lan": "LANGUAGE_noAngG", "MD": "MD", "phys": "PHYSICS", "ToM": "TOM"}
BAR_NETWORKS = ["LANGUAGE_noAngG", "MD", "TOM", "PHYSICS", "PHYSICS_Kean"]


def surface_maps(U, parcels, cortex, window):
    from nilearn import datasets, plotting
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    views = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]
    fig, axes = plt.subplots(4, 4, figsize=(8 * .9, 7.4 * .9), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=.14, right=.9, top=.93, bottom=.02, wspace=-.08, hspace=-.12)
    for i, d in enumerate(DOMAINS):
        x = np.where(cortex, U[f"contrast_{window}"][i], np.nan)
        vmax = float(np.nanquantile(np.abs(x), .99))
        net = parcels[MAP_NETWORKS[d]] > 0
        for j, (hemi, view) in enumerate(views):
            sl = slice(0, 10242) if hemi == "left" else slice(10242, None)
            ax = axes[i, j]
            plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], x[sl], hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                        axes=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax, symmetric_cbar=True,
                                        colorbar=False, threshold=None, bg_on_data=False)
            plotting.plot_surf_contours(fs[f"infl_{hemi}"], net[sl].astype(int), hemi=hemi, view=view, levels=[1],
                                        colors=["black"], axes=ax, linewidths=.8)
            for c in ax.collections:
                c.set_rasterized(True)
            if i == 0:
                ax.set_title(f"{hemi[0].upper()}H {view}", fontsize=10)
        pos = axes[i, 0].get_position()
        fig.text(.005, (pos.y0 + pos.y1) / 2, f"{DOMAIN_LABELS[d]}\n> others", fontsize=11, weight="bold",
                 color=DOMAIN_COLORS[d], va="center", ha="left")
        cax = fig.add_axes([.91, pos.y0 + .03, .012, pos.height - .06])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(.5)
    fig.text(.5, .965, "Predicted response: task domain minus the other three", ha="center", fontsize=13)
    save_fig(fig, OUT / f"univariate_maps_{window}")


def network_bars(stats, responses, selection, window):
    fig, axes = plt.subplots(1, len(BAR_NETWORKS), figsize=(11 * .85, 3.3 * .85))
    rng = np.random.default_rng(0)
    for ax, net in zip(axes, BAR_NETWORKS):
        s = stats[(stats.network == net) & (stats.selection == selection) & (stats.window == window)].iloc[0]
        r = responses[(responses.network == net) & (responses.selection == selection) & (responses.window == window)]
        per_task = r.groupby(["task", "domain"], sort=False).response.mean().reset_index()
        target = NETWORK_TARGET[net]
        for x, d in enumerate(DOMAINS):
            m, e = s[f"mean_{d}"], s[f"sem_{d}"]
            ax.bar(x, m, width=.7, color=DOMAIN_COLORS[d], alpha=.35 if d != target else .6,
                   edgecolor="black", linewidth=1.2 if d == target else .8, zorder=2)
            ax.errorbar(x, m, yerr=e, color="black", capsize=3, lw=1.2, zorder=4)
            v = per_task[per_task.domain == d].response.to_numpy()
            ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=9, color=DOMAIN_COLORS[d], edgecolor="black",
                       linewidth=.3, alpha=.8, zorder=3)
        lo, hi = min(0, per_task.response.min()), per_task.response.max()
        step = .1 * (hi - lo)
        others = sorted([d for d in DOMAINS if d != target], key=lambda d: abs(DOMAINS.index(d) - DOMAINS.index(target)))
        for k, d in enumerate(others):
            bracket(ax, DOMAINS.index(target), DOMAINS.index(d), hi + step * (.6 + 1.5 * k), step * .3,
                    stars(s[f"p_vs_{d}"]), fontsize=7)
        ax.set_ylim(lo - .05 * (hi - lo), hi + step * (.6 + 1.5 * len(others)))
        ax.set_xticks(range(4), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=30, ha="right", fontsize=10)
        ax.set_title(NETWORK_LABELS[net], fontsize=11, weight="bold")
        style_axes(ax)
    axes[0].set_ylabel("Predicted response (a.u.)", fontsize=12, labelpad=8)
    title = "whole parcels" if selection == "whole_parcel" else "held-out fROIs (top 10% per parcel)"
    fig.suptitle(f"Parcel response to the four task domains ({title})", fontsize=13)
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, OUT / f"network_bars_{selection}_{window}")


def domain_bars(dstats, selection, window, measure):
    """One panel per task domain; bars = parcel sets. measure: 'selectivity' (task response minus the set's mean
    response to the other three domains; with brackets, target set vs each other set) or 'response' (raw)."""
    sel_csv = ROOT / load_config()["paths"]["analysis"] / "univariate" / f"task_selectivity_{selection}_{window}.csv"
    per_task = pd.read_csv(sel_csv)
    if measure == "response":
        resp = pd.read_csv(ROOT / load_config()["paths"]["analysis"] / "univariate" / "parcel_responses.csv")
        resp = resp[(resp.window == window) & (resp.selection == selection)]
        per_task = resp.groupby(["network", "task", "domain"], sort=False).response.mean().reset_index()
        per_task = per_task.rename(columns={"response": "selectivity"})
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(10 * .85, 3.4 * .85))
    rng = np.random.default_rng(0)
    for ax, d in zip(axes, DOMAINS):
        s = dstats[(dstats.domain == d) & (dstats.selection == selection) & (dstats.window == window)].set_index("network")
        tgt = s.target_network.iloc[0]
        col = f"mean_{measure}"
        for x, net in enumerate(BAR_NETWORKS):
            colour = DOMAIN_COLORS[NETWORK_TARGET[net]]
            is_t = net == tgt
            ax.bar(x, s.loc[net, col], width=.7, color=colour, alpha=.6 if is_t else .3, edgecolor="black",
                   linewidth=1.2 if is_t else .8, zorder=2)
            ax.errorbar(x, s.loc[net, col], yerr=s.loc[net, f"sem_{measure}"], color="black", capsize=3, lw=1.2, zorder=4)
            v = per_task[(per_task.network == net) & (per_task.domain == d)].selectivity.to_numpy()
            ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=9, color=DOMAIN_COLORS[d], edgecolor="black",
                       linewidth=.3, alpha=.8, zorder=3)
        vals = per_task[per_task.domain == d].selectivity
        lo, hi = min(0, vals.min()), vals.max()
        step = .1 * (hi - lo)
        top = hi
        if measure == "selectivity":
            ti = BAR_NETWORKS.index(tgt)
            others = [n for n in BAR_NETWORKS if n != tgt and not (d == "phys" and n.startswith("PHYSICS"))]
            others = sorted(others, key=lambda n: abs(BAR_NETWORKS.index(n) - ti))
            for k, n in enumerate(others):
                bracket(ax, ti, BAR_NETWORKS.index(n), hi + step * (.6 + 1.5 * k), step * .3,
                        stars(s.loc[n, "p_target_gt_this"]), fontsize=7)
            top = hi + step * (.6 + 1.5 * len(others))
            ax.axhline(0, color="black", lw=.8, zorder=1)
        ax.set_ylim(lo - .05 * (hi - lo), top + .3 * step)
        ax.set_xticks(range(len(BAR_NETWORKS)), [NETWORK_LABELS[n] for n in BAR_NETWORKS], rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{DOMAIN_LABELS[d]} tasks", fontsize=11, weight="bold", color=DOMAIN_COLORS[d])
        style_axes(ax)
    axes[0].set_ylabel("Selectivity (a.u.)\ndomain − other domains" if measure == "selectivity"
                       else "Predicted response (a.u.)", fontsize=11, labelpad=6)
    where = "whole parcels" if selection == "whole_parcel" else "held-out fROIs"
    fig.suptitle(f"Which parcels prefer each task domain ({where})" if measure == "selectivity"
                 else f"Response of each parcel set to each task domain ({where})", fontsize=13)
    fig.tight_layout(w_pad=1.2)
    save_fig(fig, OUT / f"domain_bars_{measure}_{selection}_{window}")


def enrichment(spin, window):
    s = spin[spin.window == window]
    nets = BAR_NETWORKS
    E = s.pivot(index="domain", columns="network", values="enrichment").loc[DOMAINS, nets]
    Pv = s.pivot(index="domain", columns="network", values="p_spin").loc[DOMAINS, nets]
    lim = 3.0                                   # colour scale 1/8x .. 8x; values beyond are clamped
    L = np.clip(np.log2(E.clip(lower=1e-6).to_numpy()), -lim, lim)
    fig, ax = plt.subplots(figsize=(5.6 * .8, 3.6 * .8))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(DOMAINS)):
        for j in range(len(nets)):
            sig = Pv.iloc[i, j] < .05
            val = "<0.01" if E.iloc[i, j] < .01 else f"{E.iloc[i, j]:.2f}"
            ax.text(j, i, f"{val}{'*' if sig else ''}", ha="center", va="center", fontsize=8,
                    weight="bold" if sig else "normal", color="white" if abs(L[i, j]) > .6 * lim else "black")
    ax.set_xticks(range(len(nets)), [NETWORK_LABELS[n] for n in nets], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(DOMAINS)), [f"{DOMAIN_LABELS[d]} > others" for d in DOMAINS], fontsize=10)
    for t, d in zip(ax.get_yticklabels(), DOMAINS):
        t.set_color(DOMAIN_COLORS[d])
    ax.set_xticks(np.arange(len(nets) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(DOMAINS) + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03)
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    cb.set_ticks(ticks, labels=["≤1/8", "1/4", "1/2", "1", "2", "4", "≥8"])
    cb.set_label("Enrichment (log scale)", fontsize=9)
    ax.set_title("Top 10% of each map within each parcel set\n(* spin test p < .05)", fontsize=11)
    save_fig(fig, OUT / f"enrichment_{window}")


def length_language(U, parcels, cortex, window):
    tasks = pd.read_csv(ROOT / "data" / "stimuli" / "task_summary.csv")
    resp = U[f"task_means_{window}"][:, (parcels["LANGUAGE_noAngG"] > 0) & cortex].mean(1)
    rho, p = spearmanr(tasks.mean_words, resp)
    fig, ax = plt.subplots(figsize=(2.8 * .9, 2.8 * .9))
    for d in DOMAINS:
        k = tasks.domain == d
        ax.scatter(tasks.mean_words[k], resp[k], s=22, color=DOMAIN_COLORS[d], edgecolor="black", lw=.4,
                   alpha=.85, label=DOMAIN_LABELS[d], zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax.set_xlabel("Mean words per stimulus", fontsize=12)
    ax.set_ylabel("Language-parcel\nresponse (a.u.)", fontsize=12)
    ax.text(.04, .96, f"ρ = {rho:.2f}\np = {p:.0e}", transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
    save_fig(fig, OUT / f"length_language_{window}")


def main():
    apply_style()
    cfg = load_config()
    A = ROOT / cfg["paths"]["analysis"]
    U = np.load(A / "univariate.npz", allow_pickle=False)
    parcels, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    stats = pd.read_csv(A / "univariate" / "network_stats.csv")
    responses = pd.read_csv(A / "univariate" / "parcel_responses.csv")
    spin = pd.read_csv(A / "univariate" / "spin.csv")
    dstats = pd.read_csv(A / "univariate" / "domain_stats.csv")
    for window in cfg["univariate"]["windows"]:
        for selection in stats[stats.window == window].selection.unique():
            network_bars(stats, responses, selection, window)
            for measure in ["selectivity", "response"]:
                domain_bars(dstats, selection, window, measure)
        enrichment(spin, window)
        length_language(U, parcels, cortex, window)
        surface_maps(U, parcels, cortex, window)
        print("figures done:", window, flush=True)


if __name__ == "__main__":
    main()
