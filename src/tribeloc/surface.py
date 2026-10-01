"""fsaverage5 geometry: geodesic searchlights and sphere coordinates for spin tests."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from tribeloc.parcels import N_HEMI

HEMIS = [("left", slice(0, N_HEMI)), ("right", slice(N_HEMI, 2 * N_HEMI))]


def _mesh(name, hemi):
    from nilearn import datasets, surface
    m = surface.load_surf_mesh(datasets.fetch_surf_fsaverage("fsaverage5")[f"{name}_{hemi}"])
    return np.asarray(m.coordinates, float), np.asarray(m.faces)


def edge_graph(coords, faces):
    """Sparse symmetric graph of mesh edges weighted by Euclidean length."""
    e = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    d = np.linalg.norm(coords[e[:, 0]] - coords[e[:, 1]], axis=1)
    g = sparse.coo_matrix((d, (e[:, 0], e[:, 1])), shape=(len(coords),) * 2).tocsr()
    return g.maximum(g.T)


def searchlights(cortex, radius):
    """CSR-style (indptr, indices): for each cortical centre (in order of np.flatnonzero(cortex)),
    the cortical vertices of the same hemisphere within `radius` mm along edges of the midthickness mesh."""
    indptr, indices = [0], []
    for hemi, sl in HEMIS:
        pial, faces = _mesh("pial", hemi)
        white, _ = _mesh("white", hemi)
        dist = dijkstra(edge_graph((pial + white) / 2, faces), limit=radius)
        ctx = cortex[sl]
        for v in np.flatnonzero(ctx):
            members = np.flatnonzero(np.isfinite(dist[v]) & ctx) + sl.start
            indices.append(members)
            indptr.append(indptr[-1] + len(members))
    return np.array(indptr), np.concatenate(indices)


def sphere_coords():
    """Unit-sphere coordinates of the left and right fsaverage5 vertices."""
    out = []
    for hemi, _ in HEMIS:
        xyz, _ = _mesh("sphere", hemi)
        out.append(xyz / np.linalg.norm(xyz, axis=1, keepdims=True))
    return out
