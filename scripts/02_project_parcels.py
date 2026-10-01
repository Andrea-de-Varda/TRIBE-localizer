"""Verify parcel checksums, project to fsaverage5, and write an audit of voxel/vertex counts."""
import hashlib

import nibabel as nib
import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.parcels import N_HEMI, cortex_mask, label_table, project


def main():
    cfg = load_config()["parcels"]
    pdir = ROOT / "data" / "parcels"
    checks = pd.read_csv(pdir / "checksums.tsv", sep="\t").set_index("file").sha256
    cortex = cortex_mask()
    surfaces, audit = {"cortex": cortex}, []
    for net in cfg["networks"]:
        path = pdir / f"{net}.nii.gz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checks[path.name], path
        vol = np.rint(nib.load(path).get_fdata()).astype(int)
        lab = project(path)
        table = label_table(pdir / f"{net}_labels.tsv", net)
        assert set(np.unique(vol)) == {0, *table.label}, net
        for r in table.itertuples():
            assert (vol == r.label).sum() == r.n_voxels
            v = lab == r.label
            hemi_ok = v[:N_HEMI].any() if r.hemisphere == "L" else v[N_HEMI:].any()
            wrong = v[N_HEMI:].sum() if r.hemisphere == "L" else v[:N_HEMI].sum()
            audit.append(dict(network=net, label=r.label, name=r.name, hemisphere=r.hemisphere, n_voxels=r.n_voxels,
                              n_vertices=int(v.sum()), n_vertices_cortex=int((v & cortex).sum()),
                              n_vertices_wrong_hemisphere=int(wrong), present=bool(hemi_ok)))
            # Midline voxels can project to the other hemisphere; keep each parcel in its own hemisphere.
            other = slice(N_HEMI, None) if r.hemisphere == "L" else slice(0, N_HEMI)
            lab[other][lab[other] == r.label] = 0
        surfaces[net] = lab
    out = ROOT / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **surfaces)
    audit = pd.DataFrame(audit)
    audit.to_csv(ROOT / cfg["audit"], index=False)
    print(audit.to_string(index=False))
    print(f"cortex vertices: {cortex.sum()} / {cortex.size}")


if __name__ == "__main__":
    main()
