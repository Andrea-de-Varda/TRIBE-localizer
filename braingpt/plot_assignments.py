"""Assignment diagram (CPU, local): the 46 tasks at the bottom, grouped by domain, the five candidate networks at the
top, and one curve per task to the network BrainGPT assigned it to. Curves take the colour of the task's target
network, so a curve whose colour differs from the node it reaches is a misassignment.
Writes plots/assignment_diagram_<measure>_<style>.{svg,png}.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from bglib import CANDIDATES, DOMAINS, HERE
from tribeloc.plotting import DOMAIN_COLORS, DOMAIN_LABELS, apply_style, save_fig

CAND_COLORS = {**DOMAIN_COLORS, "visual": "#7f7f7f"}
CAND_LABELS = {"Lan": "Language", "MD": "Multiple\ndemand", "phys": "Physics", "ToM": "Theory\nof mind",
               "visual": "Visual\n(distractor)"}
# Short names and within-domain order as in the LLM-modularity paper (scripts/plot_accuracy_heatmap.py there).
TASK_SHORT = {
    "anaphor_gender_agreement": "Anaphor", "det_noun_agreement_irregular": "DetN-Irr", "det_noun_agreement_regular": "DetN-Reg",
    "det_noun_agreement_with_adjective": "DetN-Adj", "hypernymy": "Hyper", "npi": "NPI", "subject_verb_agreement": "S-V",
    "wug": "Wug",
    "add_sub_2op_symbolic": "Add2-Sym", "add_sub_2op_verbal": "Add2-Vrb", "add_sub_3op_symbolic": "Add3-Sym",
    "add_sub_3op_verbal": "Add3-Vrb", "mul_div_2op_symbolic": "Mul2-Sym", "mul_div_2op_verbal": "Mul2-Vrb",
    "mul_div_3op_symbolic": "Mul3-Sym", "mul_div_3op_verbal": "Mul3-Vrb", "logic_propositional_1": "PropL-NL",
    "logic_propositional_symbolic": "PropL-Sym", "logic_syllogism_1": "Syll-NL", "logic_syllogism_symbolic": "Syll-Sym",
    "code_A": "CodeA", "code_B": "CodeB", "code_conditional": "CodeCond", "code_list": "CodeList", "code_loop": "CodeLoop",
    "simple_equation": "Eq", "number_sequence": "NumSeq", "number_sorting": "NumSort",
    "phys_newton": "Newton", "phys_prost": "PROST", "physics_brightness": "Brightness", "physics_buoyancy": "Buoyancy",
    "physics_elasticity": "Elasticity", "physics_solubility": "Solubility", "physics_speed": "Speed",
    "physics_stability": "Stability", "physics_temperature": "Temperature",
    "agent": "Agent", "desires_goals": "Desires", "emotion_fewshot": "EmotionFS", "norm_appropriate": "NormApp",
    "norm_moral": "NormMoral", "primary_emotions": "PrimEmo", "secondary_emotions": "SecEmo",
    "social_interactions": "SocInt", "social_relations": "SocRel"}
GAP = 1.6           # extra horizontal space between domain groups
Y_TASK, Y_NET = 0.0, 1.0


def layout(p):
    """x position per task (domain groups in DOMAINS order, tasks in TASK_SHORT order) and per network node."""
    order = [t for t in TASK_SHORT if t in set(p.task)]
    dom = p.set_index("task").domain
    x, pos, groups = 0.0, {}, {}
    for d in DOMAINS:
        ts = [t for t in order if dom[t] == d]
        for t in ts:
            pos[t] = x
            x += 1
        groups[d] = (pos[ts[0]], pos[ts[-1]])
        x += GAP
    nodes = {d: (groups[d][0] + groups[d][1]) / 2 for d in DOMAINS}
    nodes["visual"] = x - GAP + 4.2
    return pos, groups, nodes


def curve(ax, x0, x1, colour, lw, alpha, zorder):
    y0, y1 = Y_TASK + .06, Y_NET - .09
    path = Path([(x0, y0), (x0, .55), (x1, .45), (x1, y1)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour, lw=lw, alpha=alpha, capstyle="round", zorder=zorder))


def diagram(p, measure, style):
    pos, groups, nodes = layout(p)
    fig, ax = plt.subplots(figsize=(10.5 * .85, 4.8 * .85))
    # curves: correct ones underneath, misassignments on top
    for correct in (True, False):
        for r in p[p.correct == correct].itertuples():
            curve(ax, pos[r.task], nodes[r.pick], CAND_COLORS[r.domain], 1.6 if not correct else 1.1,
                  .95 if not correct else .55, 3 if not correct else 2)
            ax.plot(pos[r.task], Y_TASK + .045, "o", ms=3.2, color=CAND_COLORS[r.domain], mec="black", mew=.3, zorder=4)
    # network nodes (box sized to the label)
    for c in CANDIDATES:
        ax.text(nodes[c], Y_NET, CAND_LABELS[c], ha="center", va="center", fontsize=9, color="white", weight="bold",
                zorder=6, linespacing=1.0, bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", fc=CAND_COLORS[c], ec="black", lw=.8,
                                    alpha=.95 if c != "visual" else .6))
    # task labels and domain brackets (below the longest label)
    dom = p.set_index("task").domain
    for t, x in pos.items():
        ax.text(x, Y_TASK - .02, TASK_SHORT[t], rotation=90, ha="center", va="top", fontsize=7, color=CAND_COLORS[dom[t]])
    y = Y_TASK - .47
    for d, (a, b) in groups.items():
        ax.plot([a - .3, a - .3, b + .3, b + .3], [y + .025, y, y, y + .025], color=DOMAIN_COLORS[d], lw=1.2)
        k = p[p.domain == d]
        ax.text((a + b) / 2, y - .03, f"{DOMAIN_LABELS[d]} tasks", ha="center", va="top", fontsize=9, weight="bold",
                color=DOMAIN_COLORS[d])
        ax.text((a + b) / 2, y - .12, f"{int(k.correct.sum())}/{len(k)} correct", ha="center", va="top", fontsize=8,
                color=DOMAIN_COLORS[d])
    ax.set_xlim(min(pos.values()) - 1, nodes["visual"] + 2)
    ax.set_ylim(Y_TASK - .68, Y_NET + .14)
    ax.axis("off")
    save_fig(fig, HERE / "plots" / f"assignment_diagram_{measure}_{style}")


def main():
    apply_style()
    picks = pd.read_csv(HERE / "results" / "picks.csv")
    for (measure, style), p in picks.groupby(["measure", "style"], sort=False):
        diagram(p.reset_index(drop=True), measure, style)
    print("assignment diagrams saved", flush=True)


if __name__ == "__main__":
    main()
