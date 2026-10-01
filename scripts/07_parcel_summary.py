"""Held-out parcel fROIs and spin tests (runs locally on the pulled results).

Writes results/analysis/parcel_heldout.csv (task x parcel), network_domain_heldout.csv
(network x domain, parcels then tasks weighted equally) and spin_tests.csv.
"""
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.group import DOMAINS, parcel_table, spin_test
from tribeloc.parcels import load as load_parcels
from tribeloc.surface import sphere_coords


def main():
    cfg = load_config()
    P, pc, sc = cfg["paths"], cfg["parcel_summary"], cfg["spin"]
    A = ROOT / P["analysis"]
    tasks = pd.read_csv(ROOT / "data" / "stimuli" / "task_summary.csv")
    parcels, cortex, audit = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    results = {t: np.load(A / "searchlight" / f"{t}.npz") for t in tasks.task}

    per = parcel_table(results, tasks, parcels, audit, cortex, pc["top_fraction"])
    per.to_csv(A / "parcel_heldout.csv", index=False)
    by_net = per.groupby(["network", "domain", "task"], sort=False)[["froi_heldout", "whole_parcel"]].mean().reset_index()
    summary = by_net.groupby(["network", "domain"], sort=False).agg(
        froi_heldout=("froi_heldout", "mean"), froi_sem=("froi_heldout", "sem"),
        whole_parcel=("whole_parcel", "mean"), n_tasks=("task", "size")).reset_index()
    summary.to_csv(A / "network_domain_heldout.csv", index=False)
    print(summary.pivot(index="network", columns="domain", values="froi_heldout")[DOMAINS].round(5).to_string())

    domain_maps = np.load(A / "domain_maps.npz")
    spheres = sphere_coords()
    rng = np.random.default_rng(sc["seed"])
    rows = []
    networks = {net: labels > 0 for net, labels in parcels.items()}
    for d in DOMAINS:
        res = spin_test(domain_maps[f"{d}_crossnobis"], cortex, networks, spheres, sc["n_rotations"], sc["top_fraction"], rng)
        rows += [dict(domain=d, network=net, **r) for net, r in res.items()]
    spin = pd.DataFrame(rows)
    spin.to_csv(A / "spin_tests.csv", index=False)
    print(spin.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
