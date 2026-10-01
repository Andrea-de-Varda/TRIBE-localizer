"""fsaverage5 sphere coordinates for spin tests."""
import numpy as np

HEMIS = ["left", "right"]


def sphere_coords():
    """Unit-sphere coordinates of the left and right fsaverage5 vertices."""
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    out = []
    for hemi in HEMIS:
        xyz = np.asarray(surface.load_surf_mesh(fs[f"sphere_{hemi}"]).coordinates, float)
        out.append(xyz / np.linalg.norm(xyz, axis=1, keepdims=True))
    return out
