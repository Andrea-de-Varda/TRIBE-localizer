"""House figure style (Andrea's scientific-figure-style) and project palettes."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ["Lan", "MD", "phys", "ToM"]
# Same names and colors as the LLM-modularity paper (scripts/plot_layer_stacked.py there).
DOMAIN_LABELS = {"Lan": "Language", "MD": "Formal", "phys": "Physics", "ToM": "Social"}
DOMAIN_COLORS = {"Lan": "#C0392B", "MD": "#2471A3", "phys": "#E67E22", "ToM": "#27AE60"}
NETWORK_LABELS = {"LANGUAGE_noAngG": "Language", "MD": "MD", "TOM": "ToM", "PHYSICS": "Physics"}
NETWORK_TARGET = {"LANGUAGE_noAngG": "Lan", "MD": "MD", "TOM": "ToM", "PHYSICS": "phys"}
NS_FILL, NS_EDGE = "#cccccc", "#999999"


def apply_style():
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "DejaVu Sans"


def style_axes(ax, lw=1.5):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(lw)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def bracket(ax, x1, x2, y, h, text, fontsize=8):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def save_fig(fig, path_stem, dpi=300):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def place_brackets(tops, pairs, scale):
    """Heights for significance brackets. tops: top of each bar's error bar (index = x); pairs: [(x1, x2), ...];
    scale: data range used for spacing. Brackets are placed shortest first; each sits just above the bars it
    spans and above every already-placed bracket whose span overlaps it, so a long bracket is not pushed up by a
    tall bar it does not reach. Returns the bracket heights (in input order) and the top of the highest label."""
    gap, h, label = .05 * scale, .025 * scale, .09 * scale
    order = sorted(range(len(pairs)), key=lambda i: abs(pairs[i][1] - pairs[i][0]))
    ys, placed = [None] * len(pairs), []
    for i in order:
        a, b = sorted(pairs[i])
        base = max(tops[a:b + 1])
        for (pa, pb), y in placed:
            if pa <= b and a <= pb:              # spans overlap (sharing an end bar counts)
                base = max(base, y + h + label)
        ys[i] = base + gap
        placed.append(((a, b), ys[i]))
    return ys, max(y + h + label for y in ys) if ys else max(tops)
