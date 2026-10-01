"""Localizer contrasts, fROIs and localizer validation (CPU; needs the localizer shards).

Outputs in results/analysis/localizers/:
  contrasts.npz        t map (all stimuli) of every localizer, per window
  frois.npz            fROI vertex indices, key "<scheme>__<network>__<parcel>"; scheme "task" uses the standard
                       TowerLoc-style physics localizer, scheme "content" the content-matched one (other networks
                       identical in both schemes)
  validation.csv       per parcel: whole-parcel effect and split-half held-out fROI effect of each localizer in
                       its own network, and of every localizer in every network (cross-localizer)
  spin.csv             top 10% of each localizer t map vs each parcel set (enrichment, spin p)
  froi_overlap.csv     shared vertices between networks' fROIs
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.baseline import stimulus_baselines
from tribeloc.froi import define_frois, heldout_effect, localizer_t
from tribeloc.group import spin_test
from tribeloc.inference import N_VERTICES, assign_shards
from tribeloc.parcels import load as load_parcels
from tribeloc.surface import sphere_coords


def load_localizer_predictions(table, windows, shard_dir, shard_size):
    shards = assign_shards(table, shard_size)
    pos = {s: i for i, s in enumerate(table.stim_id)}
    X = {w: np.full((len(table), N_VERTICES), np.nan, np.float32) for w in windows}
    for k in np.unique(shards):
        z = np.load(Path(shard_dir) / f"shard_{k:04d}.npz")
        idx = np.array([pos[s] for s in z["stim_id"]])
        for w in windows:
            X[w][idx] = z[w]
    assert all(np.isfinite(x).all() for x in X.values()), "localizer shards incomplete"
    return X


def main():
    cfg = load_config()
    lc, pc, sc = cfg["localizers"], cfg["parcel_summary"], cfg["spin"]
    table = pd.read_csv(ROOT / lc["output"])
    parcels, cortex, audit = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    X = load_localizer_predictions(table, lc["windows"], ROOT / lc["shards"], cfg["tribe"]["shard_size"])
    if cfg["baseline"]["subtract"]:                                 # relative to the window-matched no-input baseline
        z = np.load(ROOT / cfg["baseline"]["path"])
        base = stimulus_baselines(table, z["timecourse"], z["times"], {w: cfg["windows"][w] for w in lc["windows"]})
        X = {w: (x - base[w]).astype(np.float32) for w, x in X.items()}
    out = ROOT / cfg["paths"]["analysis"] / "localizers"
    out.mkdir(parents=True, exist_ok=True)

    specs = {f"{net}": spec for net, spec in lc["contrasts"].items()}
    specs["PHYSICS_content"] = lc["physics_alternative"]
    rows_of = {name: table[table.localizer == s["localizer"]].reset_index(drop=True) for name, s in specs.items()}
    X_of = {name: {w: X[w][(table.localizer == s["localizer"]).to_numpy()] for w in lc["windows"]} for name, s in specs.items()}

    # t maps (all stimuli), every localizer and window
    tmaps = {f"{name}__{w}": localizer_t(X_of[name][w], rows_of[name], s) for name, s in specs.items() for w in lc["windows"]}
    np.savez_compressed(out / "contrasts.npz", **{k: v.astype(np.float32) for k, v in tmaps.items()})

    # fROIs: selection window = first localizer window ("full")
    w0 = lc["windows"][0]
    frois = {}
    for scheme in ["task", "content"]:
        for net in lc["contrasts"]:
            name = "PHYSICS_content" if (net == "PHYSICS" and scheme == "content") else net
            labs = audit[audit.network == net]
            for lab, v in define_frois(tmaps[f"{name}__{w0}"], parcels[net], cortex, labs.label, pc["top_fraction"]).items():
                frois[f"{scheme}__{net}__{labs.set_index('label').loc[lab, 'name']}"] = v
    np.savez_compressed(out / "frois.npz", **frois)

    # validation: each localizer in each network (own network = diagonal), whole parcel and held-out fROI
    val = []
    for name, s in specs.items():
        for net in lc["contrasts"]:
            for r in audit[audit.network == net].itertuples():
                v = np.flatnonzero((parcels[net] == r.label) & cortex)
                for w in lc["windows"]:
                    x, rows = X_of[name][w], rows_of[name]
                    whole = x[:, v].mean(1)
                    tgt, ctl = (rows.condition == s["target"]).to_numpy(), (rows.condition == s["control"]).to_numpy()
                    eff, t_ho = heldout_effect(x, rows, s, v, pc["top_fraction"])
                    val.append(dict(localizer=name, network=net, own=(name.split("_content")[0] == net), parcel=r.name,
                                    window=w, whole_parcel_effect=float(whole[tgt].mean() - whole[ctl].mean()),
                                    heldout_froi_effect=eff, heldout_froi_t=t_ho))
    val = pd.DataFrame(val)
    val.to_csv(out / "validation.csv", index=False)
    own = val[val.own & (val.window == w0)]
    print(own.groupby(["localizer", "network"]).agg(
        whole=("whole_parcel_effect", "mean"), heldout=("heldout_froi_effect", "mean"),
        parcels_positive=("heldout_froi_t", lambda t: f"{(t > 2).sum()}/{len(t)}")).round(4).to_string(), flush=True)

    # where does each localizer contrast fall on the cortex? (top 10% vs parcel sets, spin test)
    spheres, rng = sphere_coords(), np.random.default_rng(sc["seed"])
    networks = {net: lab > 0 for net, lab in parcels.items()}
    spin = []
    for name in specs:
        res = spin_test(tmaps[f"{name}__{w0}"], cortex, networks, spheres, sc["n_rotations"], sc["top_fraction"], rng)
        spin += [dict(localizer=name, network=n, **r) for n, r in res.items()]
    spin = pd.DataFrame(spin)
    spin.to_csv(out / "spin.csv", index=False)
    print(spin.pivot(index="localizer", columns="network", values="enrichment").round(2).to_string(), flush=True)

    # overlap between networks' fROIs (standard scheme)
    masks = {net: np.zeros(N_VERTICES, bool) for net in lc["contrasts"]}
    for k, v in frois.items():
        scheme, net, _ = k.split("__")
        if scheme == "task":
            masks[net][v] = True
    ov = [dict(network_a=a, network_b=b, shared=int((masks[a] & masks[b]).sum()), n_a=int(masks[a].sum()),
               dice=2 * (masks[a] & masks[b]).sum() / (masks[a].sum() + masks[b].sum())) for a in masks for b in masks if a < b]
    pd.DataFrame(ov).to_csv(out / "froi_overlap.csv", index=False)
    print(pd.DataFrame(ov).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
