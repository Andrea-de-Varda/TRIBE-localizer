"""Univariate contrast maps (CPU, local): each domain minus the other three (tasks weighted equally), with
task-label permutation FWE, for every window and response reference ("noinput": relative to the no-input baseline;
"raw": relative to the vertex's mean during naturalistic stimulation). Reads results/analysis/task_means.npz,
writes results/analysis/univariate.npz with task_means_<ref>_<w>, contrast_<ref>_<w>, p_fwe_<ref>_<w>.
"""
import numpy as np

from tribeloc import ROOT, load_config
from tribeloc.group import DOMAINS, univariate_permutation
from tribeloc.parcels import load as load_parcels


def main():
    cfg = load_config()
    uc, P = cfg["univariate"], cfg["paths"]
    T = np.load(ROOT / P["analysis"] / "task_means.npz")
    domains = T["task_domains"].astype(str)
    _, cortex, _ = load_parcels(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])
    out = dict(domains=np.array(DOMAINS, dtype=str), task_domains=domains)
    for ref in cfg["baseline"]["references"]:
        rng = np.random.default_rng(uc["seed"])                     # same permutations for both references
        for w in uc["windows"]:
            m = T[f"{ref}_{w}"]
            contrast, p = univariate_permutation(m, domains, cortex, uc["n_permutations"], rng)
            out[f"task_means_{ref}_{w}"], out[f"contrast_{ref}_{w}"], out[f"p_fwe_{ref}_{w}"] = m, contrast, p
            print(ref, w, {d: int((p[i] < .05).sum()) for i, d in enumerate(DOMAINS)}, "FWE p<.05 vertices", flush=True)
    np.savez_compressed(ROOT / P["analysis"] / "univariate.npz", **out)


if __name__ == "__main__":
    main()
