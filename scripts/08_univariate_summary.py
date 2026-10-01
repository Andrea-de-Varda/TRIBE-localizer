"""Univariate parcel summaries and spin tests (runs locally on results/analysis/univariate.npz).

For every parcel set, window and selection (whole parcel; held-out fROI when item-half maps exist):
per-task responses averaged over parcels (equal weight), the target domain against the other three and
against each other domain (task-label permutations), and spin tests of each domain contrast map against
each parcel set. Writes results/analysis/univariate/{parcel_responses,network_stats,spin}.csv.
"""
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.group import DOMAINS, froi_responses, pairwise_permutation, spin_test, target_permutation
from tribeloc.parcels import load as load_parcels
from tribeloc.surface import sphere_coords


def main():
    cfg = load_config()
    pc, sc, P = cfg["parcel_summary"], cfg["spin"], cfg["paths"]
    U = np.load(ROOT / P["analysis"] / "univariate.npz")
    tasks = pd.read_csv(ROOT / "data" / "stimuli" / "task_summary.csv")   # task order = task_index
    domains = tasks.domain.to_numpy().astype(str)
    parcels, cortex, audit = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    out = ROOT / P["analysis"] / "univariate"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pc["seed"])

    rows, stats = [], []
    for w in cfg["univariate"]["windows"]:
        selections = ["whole_parcel"] + (["froi"] if f"task_means_{w}_h1" in U.files else [])
        for net, target in pc["targets"].items():
            parcel_list = audit[audit.network == net]
            vsets = [np.flatnonzero((parcels[net] == r.label) & cortex) for r in parcel_list.itertuples()]
            for sel in selections:
                if sel == "whole_parcel":
                    per_parcel = np.stack([U[f"task_means_{w}"][:, v].mean(1) for v in vsets])
                else:
                    per_parcel = np.stack([froi_responses(U[f"task_means_{w}_h1"], U[f"task_means_{w}_h2"], domains, v,
                                                          target, pc["top_fraction"]) for v in vsets])
                for r, vals in zip(parcel_list.itertuples(), per_parcel):
                    rows += [dict(window=w, selection=sel, network=net, parcel=r.name, hemisphere=r.hemisphere,
                                  task=t, domain=d, response=float(x)) for t, d, x in zip(tasks.task, domains, vals)]
                values = per_parcel.mean(0)
                eff, p = target_permutation(values, domains, target, pc["n_permutations"], rng)
                rec = dict(window=w, selection=sel, network=net, target=target, effect_vs_others=eff, p_vs_others=p)
                for d in DOMAINS:
                    rec[f"mean_{d}"] = values[domains == d].mean()
                    rec[f"sem_{d}"] = values[domains == d].std(ddof=1) / np.sqrt((domains == d).sum())
                    if d != target:
                        rec[f"effect_vs_{d}"], rec[f"p_vs_{d}"] = pairwise_permutation(values, domains, target, d, pc["n_permutations"], rng)
                stats.append(rec)
                print(w, sel, net, f"target {target}: effect {eff:.4f}, p {p:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "parcel_responses.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "network_stats.csv", index=False)

    spheres = sphere_coords()
    srng = np.random.default_rng(sc["seed"])
    networks = {net: labels > 0 for net, labels in parcels.items()}
    spin = []
    for w in cfg["univariate"]["windows"]:
        for i, d in enumerate(DOMAINS):
            res = spin_test(U[f"contrast_{w}"][i], cortex, networks, spheres, sc["n_rotations"], sc["top_fraction"], srng)
            spin += [dict(window=w, domain=d, network=net, **r) for net, r in res.items()]
    spin = pd.DataFrame(spin)
    spin.to_csv(out / "spin.csv", index=False)
    print(spin.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
