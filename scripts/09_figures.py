"""Figures for the univariate analysis -> plots/ (SVG with editable text + PNG).

  univariate_maps       each domain minus the other three, on the inflated surface, with the target parcels outlined
  network_bars_<sel>    predicted response of each parcel set to the four task domains
  domain_bars_<sel>     transposed: for each task domain, the selectivity of each parcel set for it (and the raw response)

Units: TRIBE was trained on BOLD detrended and z-scored per vertex and run, so predictions are in z units
(SDs of the training signal). Responses are relative to the window-matched no-input baseline (TRIBE's
prediction with zero text features), the computational counterpart of a no-stimulus baseline, not human rest.
  enrichment            overlap of each domain map's top 10% with each parcel set, with spin-test significance
  length_language       language-parcel response vs stimulus length across the 46 tasks (the Language confound)
  main_figure_<measure> A: contrast maps with target parcels outlined; B: per domain, every network's fROI
                        response to it (response) or selectivity for it (selectivity), or the target
                        network's fROI response to the four domains (network)
  localizer_maps        t map of each text localizer contrast, with its network's parcels outlined
  localizer_validation  split-half held-out effect of each localizer in its own network's parcels
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

from tribeloc import ROOT, load_config
from tribeloc.parcels import load as load_parcels
from tribeloc.plotting import (DOMAIN_COLORS, DOMAIN_LABELS, DOMAINS, NETWORK_LABELS, NETWORK_TARGET, apply_style,
                               bracket, place_brackets, save_fig, stars, style_axes)

OUT = ROOT / "plots"
MAP_NETWORKS = {"Lan": "LANGUAGE_noAngG", "MD": "MD", "phys": "PHYSICS", "ToM": "TOM"}
BAR_NETWORKS = ["LANGUAGE_noAngG", "MD", "TOM", "PHYSICS"]
Z = "Predicted BOLD (z)"         # set per response reference in main()
UNI = ROOT / "results" / "analysis" / "univariate"   # set per response reference in main()
Z_LABELS = {"noinput": "Predicted BOLD vs no input (z)", "raw": "Predicted BOLD (z)"}
TALL = 1.4                      # bar figures 40% taller than the first version
DOT_ALPHA = {False: .4, True: .9}   # task dots: faded on non-target bars, strong on the target bar (same colour as the bar)


VIEWS = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]


def surface_rows(maps, outlines, labels, colours, cbar_label, stem):
    """One row per map: four views on the inflated surface, outline drawn, symmetric scale at the 99th
    percentile of |value| per row, with a labelled colour bar."""
    from nilearn import datasets, plotting
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    n = len(maps)
    fig, axes = plt.subplots(n, 4, figsize=(8.6 * .9, 1.85 * n * .9), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=.2, right=.9, top=.96, bottom=.02, wspace=-.08, hspace=-.12)
    for i, (x, outline, label, colour) in enumerate(zip(maps, outlines, labels, colours)):
        vmax = float(np.nanquantile(np.abs(x), .99))
        for j, (hemi, view) in enumerate(VIEWS):
            sl = slice(0, 10242) if hemi == "left" else slice(10242, None)
            ax = axes[i, j]
            plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], x[sl], hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                        axes=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax, symmetric_cbar=True,
                                        colorbar=False, threshold=None, bg_on_data=False)
            plotting.plot_surf_contours(fs[f"infl_{hemi}"], outline[sl].astype(int), hemi=hemi, view=view, levels=[1],
                                        colors=["black"], axes=ax, linewidths=.8)
            for c in ax.collections:
                c.set_rasterized(True)
            if i == 0:
                ax.set_title(f"{hemi[0].upper()}H {view}", fontsize=10)
        pos = axes[i, 0].get_position()
        fig.text(.005, (pos.y0 + pos.y1) / 2, label, fontsize=11, weight="bold", color=colour, va="center", ha="left")
        cax = fig.add_axes([.91, pos.y0 + .03 * 4 / n, .012, pos.height - .06 * 4 / n])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax)), cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(.5)
        cb.set_label(cbar_label, fontsize=8)
    save_fig(fig, OUT / stem)


def localizer_maps(parcels, cortex):
    A = ROOT / load_config()["paths"]["analysis"] / "localizers"
    if not (A / "contrasts.npz").exists():
        return
    C = np.load(A / "contrasts.npz")
    rows = [("LANGUAGE_noAngG", "LANGUAGE_noAngG", "Language\nsentences >\nnonwords", "Lan"),
            ("MD", "MD", "MD\nhard > easy\narithmetic", "MD"),
            ("TOM", "TOM", "ToM\nfalse belief >\nfalse photo", "ToM"),
            ("PHYSICS", "PHYSICS", "Physics (task)\nphysics > colour\nquestion", "phys"),
            ("PHYSICS_content", "PHYSICS", "Physics (content)\nphysical > colour\ndescription", "phys")]
    maps = [np.where(cortex, C[f"{name}__full"], np.nan) for name, *_ in rows]
    surface_rows(maps, [parcels[net] > 0 for _, net, *_ in rows], [r[2] for r in rows],
                 [DOMAIN_COLORS[r[3]] for r in rows], "t (localizer contrast)", "localizer_maps_full")


def localizer_validation():
    A = ROOT / load_config()["paths"]["analysis"] / "localizers"
    if not (A / "validation.csv").exists():
        return
    v = pd.read_csv(A / "validation.csv")
    v = v[v.own & (v.window == "full")]
    order = [("LANGUAGE_noAngG", "Language", "Lan"), ("MD", "MD", "MD"), ("TOM", "ToM", "ToM"),
             ("PHYSICS", "Physics\n(task)", "phys"), ("PHYSICS_content", "Physics\n(content)", "phys")]
    fig, ax = plt.subplots(figsize=(4.2 * .9, 3.0 * .9 * TALL))
    rng = np.random.default_rng(0)
    for x, (name, label, d) in enumerate(order):
        q = v[v.localizer == name]
        for k, (col, dx, alpha) in enumerate([("whole_parcel_effect", -.18, .3), ("heldout_froi_effect", .18, .7)]):
            m, e = q[col].mean(), q[col].std(ddof=1) / np.sqrt(len(q))
            ax.bar(x + dx, m, width=.34, color=DOMAIN_COLORS[d], alpha=alpha, edgecolor="black", lw=.8, zorder=2)
            ax.errorbar(x + dx, m, yerr=e, color="black", capsize=2.5, lw=1.1, zorder=4)
            ax.scatter(x + dx + rng.uniform(-.08, .08, len(q)), q[col], s=7, color=DOMAIN_COLORS[d], edgecolor="black",
                       lw=.3, alpha=.8, zorder=3)
    ax.axhline(0, color="black", lw=.8, zorder=1)
    ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=9)
    ax.set_ylabel("Localizer effect (z)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=.3, edgecolor="black", label="whole parcel"),
                       Patch(facecolor="gray", alpha=.7, edgecolor="black", label="fROI, held-out half")],
              frameon=False, fontsize=8, loc="upper right")
    style_axes(ax)
    save_fig(fig, OUT / "localizer_validation_full")


def brain_row(axes4, x, parcel_mask, vmax, fs):
    """Four inflated-surface views of x with the parcels outlined."""
    from nilearn import plotting
    for ax, (hemi, view) in zip(axes4, VIEWS):
        sl = slice(0, 10242) if hemi == "left" else slice(10242, None)
        plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], x[sl], hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                    axes=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax, symmetric_cbar=True,
                                    colorbar=False, threshold=None, bg_on_data=False)
        plotting.plot_surf_contours(fs[f"infl_{hemi}"], parcel_mask[sl].astype(int), hemi=hemi, view=view, levels=[1],
                                    colors=["black"], axes=ax, linewidths=.8)
        for c in ax.collections:
            c.set_rasterized(True)


def view_labels(fig, axes4):
    """View names just above the first row (3D axes leave a lot of empty space at the top)."""
    for ax, (hemi, view) in zip(axes4, VIEWS):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 - .1 * pos.height, f"{hemi[0].upper()}H {view}", ha="center", fontsize=10)


def colorbar(fig, pos, vmax, label, x=.91):
    cax = fig.add_axes([x, pos.y0 + .2 * pos.height, .011, .6 * pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax)), cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(.5)
    cb.set_label(label, fontsize=8)


def surface_maps(U, parcels, cortex, window):
    from nilearn import datasets
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    fig, axes = plt.subplots(4, 4, figsize=(8 * .9, 6.6 * .9), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=.14, right=.9, top=1.02, bottom=.0, wspace=-.08, hspace=-.25)
    for i, d in enumerate(DOMAINS):
        x = np.where(cortex, U[f"contrast_{window}"][i], np.nan)
        vmax = float(np.nanquantile(np.abs(x), .99))
        net = MAP_NETWORKS[d]
        brain_row(axes[i], x, parcels[net] > 0, vmax, fs)
        pos = axes[i, 0].get_position()
        fig.text(.005, (pos.y0 + pos.y1) / 2, f"{DOMAIN_LABELS[d]}\n> others", fontsize=11, weight="bold",
                 color=DOMAIN_COLORS[d], va="center", ha="left")
        colorbar(fig, pos, vmax, "Δ predicted BOLD (z)")
    view_labels(fig, axes[0])
    save_fig(fig, OUT / f"univariate_maps_{window}")


def network_panel(ax, s, per_task, net, lo, hi, rng, title=True):
    """Bars: one parcel set's response to the four task domains (mean over tasks ± SEM, dots = tasks);
    brackets: target domain vs each other domain (one-sided task-label permutation p)."""
    target = NETWORK_TARGET[net]
    for x, d in enumerate(DOMAINS):
        m, e = s[f"mean_{d}"], s[f"sem_{d}"]
        ax.bar(x, m, width=.7, color=DOMAIN_COLORS[d], alpha=.35 if d != target else .6,
               edgecolor="black", linewidth=1.2 if d == target else .8, zorder=2)
        ax.errorbar(x, m, yerr=e, color="black", capsize=3, lw=1.2, zorder=4)
        v = per_task[per_task.domain == d].response.to_numpy()
        ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=9, color=DOMAIN_COLORS[d], edgecolor="black",
                   linewidth=.3, alpha=DOT_ALPHA[d == target], zorder=3)
    others = [d for d in DOMAINS if d != target]
    tops = [s[f"mean_{d}"] + s[f"sem_{d}"] for d in DOMAINS]
    pairs = [(DOMAINS.index(target), DOMAINS.index(d)) for d in others]
    ys, top = place_brackets(tops, pairs, hi - lo)
    for (x1, x2), y, d in zip(pairs, ys, others):
        bracket(ax, x1, x2, y, .025 * (hi - lo), stars(s[f"p_vs_{d}"]), fontsize=7)
    ax.set_ylim(lo - .05 * (hi - lo), max(top, hi + .03 * (hi - lo)))
    ax.set_xticks(range(4), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=30, ha="right", fontsize=10)
    if title:
        ax.set_title(f"{NETWORK_LABELS[net]} parcels", fontsize=11, weight="bold")
    style_axes(ax)
    return ax.get_ylim()


def network_bars(stats, responses, selection, window):
    fig, axes = plt.subplots(1, len(BAR_NETWORKS), figsize=(8.8 * .85, 3.3 * .85 * TALL), sharey=True)
    rng = np.random.default_rng(0)
    sub = responses[(responses.selection == selection) & (responses.window == window)]
    per_all = sub.groupby(["network", "task", "domain"], sort=False).response.mean().reset_index()
    lo, hi = min(0, per_all.response.min()), per_all.response.max()
    lims = []
    for ax, net in zip(axes, BAR_NETWORKS):
        s = stats[(stats.network == net) & (stats.selection == selection) & (stats.window == window)].iloc[0]
        lims.append(network_panel(ax, s, per_all[per_all.network == net], net, lo, hi, rng))
    axes[0].set_ylim(lims[0][0], max(l[1] for l in lims))
    axes[0].set_ylabel(Z, fontsize=12, labelpad=8)
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, OUT / f"network_bars_{selection}_{window}")


def main_figure(U, parcels, cortex, dstats, window, selection="froi_content", measure="response", stats=None,
                responses=None):
    """A: each domain's contrast map (whole cortex, target parcels outlined). B, next to each map:
    measure="response": every network's fROI response to that domain's tasks;
    measure="selectivity": every network's fROI selectivity for that domain;
    measure="network": the target network's fROI response to the four task domains (needs stats, responses).
    Brackets compare the target with each alternative. Each bar panel has its own y-range."""
    from nilearn import datasets
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    if measure == "network":
        sub = responses[(responses.selection == selection) & (responses.window == window)]
        per_all = sub.groupby(["network", "task", "domain"], sort=False).response.mean().reset_index()
    else:
        vals = domain_values(selection, window, measure)
    fig = plt.figure(figsize=(12.2 * .9, 7.4 * .9))
    gb = fig.add_gridspec(4, 4, left=.1, right=.64, top=1.0, bottom=.02, wspace=-.06, hspace=-.12)
    gr = fig.add_gridspec(4, 1, left=.8, right=.955, top=.955, bottom=.085, hspace=.38)
    rng = np.random.default_rng(0)
    first_row, bar_axes = None, []
    for i, d in enumerate(DOMAINS):
        net = MAP_NETWORKS[d]
        axes4 = [fig.add_subplot(gb[i, j], projection="3d") for j in range(4)]
        first_row = first_row or axes4
        x = np.where(cortex, U[f"contrast_{window}"][i], np.nan)
        vmax = float(np.nanquantile(np.abs(x), .99))
        brain_row(axes4, x, parcels[net] > 0, vmax, fs)
        pos = axes4[0].get_position()
        fig.text(.005, (pos.y0 + pos.y1) / 2, f"{DOMAIN_LABELS[d]}\n> others", fontsize=11, weight="bold",
                 color=DOMAIN_COLORS[d], va="center", ha="left")
        colorbar(fig, pos, vmax, "Δ predicted BOLD (z)", x=.648)
        bax = fig.add_subplot(gr[i])
        bar_axes.append(bax)
        if measure == "network":
            s = stats[(stats.network == net) & (stats.selection == selection) & (stats.window == window)].iloc[0]
            pt = per_all[per_all.network == net]
            network_panel(bax, s, pt, net, min(0, pt.response.min()), pt.response.max(), rng, title=False)
            side, colour = f"{NETWORK_LABELS[net]} fROIs", "black"
        else:
            s = dstats[(dstats.domain == d) & (dstats.selection == selection) & (dstats.window == window)].set_index("network")
            domain_panel(bax, s, vals, d, measure, None, None, rng, title=False)
            side, colour = f"{DOMAIN_LABELS[d]} tasks", DOMAIN_COLORS[d]
        bax.text(1.03, .5, side, transform=bax.transAxes, rotation=270, va="center", ha="left",
                 fontsize=10, weight="bold", color=colour)
        bax.set_ylabel("")
        bax.tick_params(labelsize=8)
        if i < 3:
            bax.set_xticklabels([])
    bar_axes[-1].set_xlabel("Task domain" if measure == "network" else "fROIs", fontsize=10)
    fig.text(.752, .52, SEL_LABEL if measure == "selectivity" else Z, rotation=90, ha="center", va="center", fontsize=11)
    view_labels(fig, first_row)
    fig.text(.005, .985, "A", fontsize=16, weight="bold", va="top")
    fig.text(.735, .985, "B", fontsize=16, weight="bold", va="top")
    save_fig(fig, OUT / f"main_figure_{measure}_{selection}_{window}")


def domain_values(selection, window, measure):
    """Per-task values per parcel set: selectivity, or raw response."""
    A = UNI
    if measure == "selectivity":
        per_task = pd.read_csv(A / f"task_selectivity_{selection}_{window}.csv").rename(columns={"selectivity": "value"})
    else:
        resp = pd.read_csv(A / "parcel_responses.csv")
        resp = resp[(resp.window == window) & (resp.selection == selection)]
        per_task = resp.groupby(["network", "task", "domain"], sort=False).response.mean().reset_index()
        per_task = per_task.rename(columns={"response": "value"})
    return per_task[per_task.network.isin(BAR_NETWORKS)]


def domain_panel(ax, s, per_task, d, measure, lo, hi, rng, title=True):
    """Bars: each parcel set's selectivity for (or response to) domain d; brackets compare the target set with
    each other set (one-sided paired sign-flip across the domain's tasks, on the plotted measure).
    lo/hi None: y-range from this domain's values only."""
    if lo is None:
        v = per_task[per_task.domain == d].value
        lo, hi = min(0, v.min()), v.max()
    tgt = s.target_network.iloc[0]
    col = f"mean_{measure}"
    for x, net in enumerate(BAR_NETWORKS):
        is_t = net == tgt
        ax.bar(x, s.loc[net, col], width=.7, color=DOMAIN_COLORS[NETWORK_TARGET[net]], alpha=.6 if is_t else .3,
               edgecolor="black", linewidth=1.2 if is_t else .8, zorder=2)
        ax.errorbar(x, s.loc[net, col], yerr=s.loc[net, f"sem_{measure}"], color="black", capsize=3, lw=1.2, zorder=4)
        v = per_task[(per_task.network == net) & (per_task.domain == d)].value.to_numpy()
        ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=9, color=DOMAIN_COLORS[NETWORK_TARGET[net]],
                   edgecolor="black", linewidth=.3, alpha=DOT_ALPHA[is_t], zorder=3)
    pcol = "p_target_gt_this" if measure == "selectivity" else "p_resp_target_gt_this"
    ti = BAR_NETWORKS.index(tgt)
    others = [n for n in BAR_NETWORKS if n != tgt]
    tops = [s.loc[n, col] + s.loc[n, f"sem_{measure}"] for n in BAR_NETWORKS]
    pairs = [(ti, BAR_NETWORKS.index(n)) for n in others]
    ys, top = place_brackets(tops, pairs, hi - lo)
    for (x1, x2), y, n in zip(pairs, ys, others):
        bracket(ax, x1, x2, y, .025 * (hi - lo), stars(s.loc[n, pcol]), fontsize=7)
    ax.axhline(0, color="black", lw=.8, zorder=1)
    ax.set_ylim(lo - .05 * (hi - lo), max(top, hi + .03 * (hi - lo)))
    ax.set_xticks(range(len(BAR_NETWORKS)), [NETWORK_LABELS[n] for n in BAR_NETWORKS], rotation=30, ha="right", fontsize=10)
    if title:
        ax.set_title(f"{DOMAIN_LABELS[d]} tasks", fontsize=11, weight="bold", color=DOMAIN_COLORS[d])
    style_axes(ax)
    return ax.get_ylim()


SEL_LABEL = "Selectivity (z)"


def domain_bars(dstats, selection, window, measure):
    """One panel per task domain; bars = parcel sets (selectivity with brackets, or raw response)."""
    per_task = domain_values(selection, window, measure)
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(8.8 * .85, 3.4 * .85 * TALL), sharey=True)
    rng = np.random.default_rng(0)
    lo, hi = min(0, per_task.value.min()), per_task.value.max()
    lims = []
    for ax, d in zip(axes, DOMAINS):
        s = dstats[(dstats.domain == d) & (dstats.selection == selection) & (dstats.window == window)].set_index("network")
        lims.append(domain_panel(ax, s, per_task, d, measure, lo, hi, rng))
    axes[0].set_ylim(lims[0][0], max(l[1] for l in lims))
    axes[0].set_ylabel(SEL_LABEL if measure == "selectivity" else Z, fontsize=11, labelpad=6)
    fig.supxlabel("Parcels", fontsize=11, y=.02)
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
    cb.set_label("Enrichment of the top 10% (log scale)", fontsize=9)
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
    ax.set_ylabel("Language-parcel " + Z[0].lower() + Z[1:], fontsize=10)
    ax.text(.04, .96, f"ρ = {rho:.2f}\np = {p:.0e}", transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
    save_fig(fig, OUT / f"length_language_{window}")


def main():
    global OUT, UNI, Z
    apply_style()
    cfg = load_config()
    A = ROOT / cfg["paths"]["analysis"]
    U = np.load(A / "univariate.npz", allow_pickle=False)
    parcels, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    root_out = OUT
    for ref in cfg["baseline"]["references"]:
        OUT, UNI, Z = root_out / ref, A / "univariate" / ref, Z_LABELS[ref]
        Uref = {f"{k}_{w}": U[f"{k}_{ref}_{w}"] for k in ["task_means", "contrast", "p_fwe"] for w in cfg["univariate"]["windows"]}
        stats = pd.read_csv(UNI / "network_stats.csv")
        responses = pd.read_csv(UNI / "parcel_responses.csv")
        spin = pd.read_csv(UNI / "spin.csv")
        dstats = pd.read_csv(UNI / "domain_stats.csv")
        for window in cfg["univariate"]["windows"]:
            for selection in stats[stats.window == window].selection.unique():
                network_bars(stats, responses, selection, window)
                for measure in ["selectivity", "response"]:
                    domain_bars(dstats, selection, window, measure)
            enrichment(spin, window)
            length_language(Uref, parcels, cortex, window)
            surface_maps(Uref, parcels, cortex, window)
            if "froi_content" in set(stats.selection):
                for measure in ["response", "selectivity", "network"]:
                    main_figure(Uref, parcels, cortex, dstats, window, measure=measure, stats=stats, responses=responses)
            print("figures done:", ref, window, flush=True)
    OUT = root_out
    localizer_maps(parcels, cortex)
    localizer_validation()


if __name__ == "__main__":
    main()
