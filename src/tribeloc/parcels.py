"""Parcels and cortex mask on TRIBE's fsaverage5 surface (20,484 vertices: 10,242 left, then right)."""
import numpy as np
import pandas as pd

N_HEMI = 10242


def project(nifti_path):
    """Integer labels per fsaverage5 vertex: neuromaps registration-fusion MNI152 -> fsaverage, nearest neighbour."""
    from neuromaps import transforms
    surf = transforms.mni152_to_fsaverage(str(nifti_path), fsavg_density="10k", method="nearest")
    return np.rint(np.concatenate([h.agg_data() for h in surf])).astype(int)


def cortex_mask():
    """Destrieux fsaverage5 vertices excluding 'Unknown' and 'Medial_wall'."""
    from nilearn import datasets
    a = datasets.fetch_atlas_surf_destrieux()
    ann = np.r_[a.map_left, a.map_right]
    excluded = a.lut.loc[a.lut.name.str.lower().isin(["unknown", "medial_wall"]), "index"].to_numpy()
    return ~np.isin(ann, excluded)


def label_table(labels_tsv, network):
    t = pd.read_csv(labels_tsv, sep="\t")
    t.insert(0, "network", network)
    return t[["network", "label", "name", "hemisphere", "n_voxels"]]


def load(npz_path, audit_path):
    """(dict network -> labels, cortex mask, audit table) as saved by scripts/02_project_parcels.py."""
    data = np.load(npz_path)
    labels = {k: data[k] for k in data.files if k != "cortex"}
    return labels, data["cortex"], pd.read_csv(audit_path)
