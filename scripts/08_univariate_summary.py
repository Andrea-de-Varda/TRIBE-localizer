"""Univariate parcel summaries and spin tests (runs locally on results/analysis/univariate.npz).

For every parcel set, window and selection (whole parcel; held-out fROI when item-half maps exist):
per-task responses averaged over parcels (equal weight), the target domain against the other three and
against each other domain (task-label permutations), and spin tests of each domain contrast map against
each parcel set. Transposed view (domain_stats.csv): for each domain, the selectivity of its tasks (task response
minus the parcel set's mean response to the other three domains) in every parcel set, and the target parcel set
against each other set (paired sign-flip test across the domain's tasks).
Writes results/analysis/univariate/{parcel_responses,network_stats,domain_stats,spin}.csv.
"""
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.group import (DOMAINS, froi_responses, paired_signflip, pairwise_permutation, spin_test, target_permutation,
                            task_selectivity)
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
    responses = pd.DataFrame(rows)
    responses.to_csv(out / "parcel_responses.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "network_stats.csv", index=False)

    # Transposed view: for each domain, which parcel set is most selective for it.
    target_net = {}
    for net, d in pc["targets"].items():
        target_net.setdefault(d, net)                    # first listed set per domain (PHYSICS_Kean for phys)
    target_net["phys"] = "PHYSICS"                       # the Casto set is primary for the domain view, Kean shown alongside
    dstats = []
    for (w, sel), r in responses.groupby(["window", "selection"], sort=False):
        per = r.groupby(["network", "task"], sort=False).response.mean().unstack("network")   # tasks x sets
        per = per.loc[tasks.task]
        selv = pd.DataFrame({n: task_selectivity(per[n].to_numpy(), domains) for n in per.columns}, index=per.index)
        for d in DOMAINS:
            k = domains == d
            tgt = target_net[d]
            for n in per.columns:
                rec = dict(window=w, selection=sel, domain=d, network=n, target_network=tgt,
                           mean_selectivity=selv[n][k].mean(), sem_selectivity=selv[n][k].std(ddof=1) / np.sqrt(k.sum()),
                           mean_response=per[n][k].mean(), sem_response=per[n][k].std(ddof=1) / np.sqrt(k.sum()))
                if n != tgt:
                    rec["diff_target_minus_this"], rec["p_target_gt_this"] = paired_signflip(
                        (selv[tgt] - selv[n])[k].to_numpy(), pc["n_permutations"], rng)
                dstats.append(rec)
        selv.assign(domain=domains).reset_index().melt(id_vars=["task", "domain"], var_name="network", value_name="selectivity") \
            .assign(window=w, selection=sel).to_csv(out / f"task_selectivity_{sel}_{w}.csv", index=False)
    dstats = pd.DataFrame(dstats)
    dstats.to_csv(out / "domain_stats.csv", index=False)
    print(dstats[dstats.selection == "whole_parcel"].round(4).to_string(index=False), flush=True)

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
