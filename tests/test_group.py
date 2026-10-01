import numpy as np
import pandas as pd
import pytest

from tribeloc.group import (domain_contrast, fwe_p, heldout_froi, overlap_fraction, parcel_table, random_rotations,
                            spin_maps, spin_test, top_vertices, univariate_permutation)
from tribeloc.surface import edge_graph, searchlights, sphere_coords


def test_fwe_p():
    null = np.array([[0.0, 1.0], [0.0, 3.0], [2.0, 0.0]])  # maxima 1, 3, 2
    assert np.allclose(fwe_p(np.array([0.5, 2.0, 5.0]), null), [4 / 4, 3 / 4, 1 / 4])


def test_domain_contrast_equal_task_weights():
    maps = np.array([[1.0], [3.0], [10.0], [0.0], [0.0]])
    domains = np.array(["Lan", "Lan", "MD", "phys", "ToM"])
    c = domain_contrast(maps, domains)
    assert np.allclose(c[:, 0], [2 - 10 / 3, 10 - 2 / 3, 0 - 12 / 3, 0 - 12 / 3])


def test_univariate_permutation_finds_domain_specific_vertex():
    rng = np.random.default_rng(0)
    domains = np.repeat(["Lan", "MD", "phys", "ToM"], 8)
    maps = rng.normal(size=(32, 50)) * 0.1
    maps[domains == "MD", 7] += 2
    cortex = np.ones(52, bool)
    cortex[[0, 51]] = False
    full = np.zeros((32, 52))
    full[:, cortex] = maps
    contrast, p = univariate_permutation(full, domains, cortex, 500, rng)
    assert np.isnan(contrast[:, 0]).all() and np.isnan(p[:, 51]).all()
    assert p[1, 8] < 0.01 and np.nanmin(np.delete(p[1], 8)) > 0.05


def test_top_vertices_and_heldout_froi():
    v = np.arange(10)
    h1 = np.arange(10.0)
    h2 = np.arange(10.0)[::-1]
    assert list(top_vertices(h1, v, 0.2)) == [8, 9]
    # top 2 of h1 are vertices 8, 9 (h2 values 1, 0); top 2 of h2 are 0, 1 (h1 values 0, 1)
    assert heldout_froi(h1, h2, v, 0.2) == pytest.approx((0.5 + 0.5) / 2)
    assert heldout_froi(h1, h1, v, 0.2) == pytest.approx(8.5)


def test_parcel_table():
    cortex = np.array([True, True, True, False])
    parcels = {"NET": np.array([1, 1, 0, 1])}
    audit = pd.DataFrame(dict(network=["NET"], label=[1], name=["p1"], hemisphere=["L"]))
    tasks = pd.DataFrame(dict(task=["t"], domain=["MD"]))
    res = {"t": dict(crossnobis=np.array([1.0, 3.0, 9.0, 99.0]), crossnobis_h1=np.array([1.0, 2.0, 0, 0]),
                     crossnobis_h2=np.array([4.0, 2.0, 0, 0]))}
    t = parcel_table(res, tasks, parcels, audit, cortex, 0.5)
    assert t.n_vertices.item() == 2 and t.whole_parcel.item() == 2.0
    assert t.froi_heldout.item() == pytest.approx((2.0 + 1.0) / 2)


def test_edge_graph_is_symmetric_with_lengths():
    coords = np.array([[0.0, 0, 0], [3, 0, 0], [0, 4, 0]])
    g = edge_graph(coords, np.array([[0, 1, 2]])).toarray()
    assert np.allclose(g, g.T) and g[1, 2] == pytest.approx(5.0)


@pytest.fixture(scope="module")
def cortex():
    from tribeloc import ROOT, load_config
    from tribeloc.parcels import load
    cfg = load_config()
    return load(ROOT / cfg["parcels"]["output"], ROOT / cfg["parcels"]["audit"])[1]


def test_searchlights_on_fsaverage5(cortex):
    indptr, indices = searchlights(cortex, 10.0)
    centres = np.flatnonzero(cortex)
    sizes = np.diff(indptr)
    assert len(sizes) == len(centres) and sizes.min() >= 5 and np.median(sizes) > 20
    for c in [0, 5000, len(centres) - 1]:
        members = indices[indptr[c]:indptr[c + 1]]
        assert centres[c] in members and cortex[members].all()
        assert len(set(members < 10242)) == 1  # same hemisphere


def test_spin_identity_and_rotation_preserves_values(cortex):
    spheres = sphere_coords()
    values = np.where(cortex, np.arange(20484.0), np.nan)
    eye = (np.eye(3)[None], np.eye(3)[None])
    assert np.array_equal(next(spin_maps(values, spheres, eye)), values, equal_nan=True)
    spun = next(spin_maps(values, spheres, random_rotations(1, np.random.default_rng(0))))
    left = spun[:10242]
    assert np.nanmax(left) < 10242  # left values stay in the left hemisphere
    assert 0.8 < np.isfinite(spun).mean() / cortex.mean() < 1.2


def test_spin_test_detects_overlap(cortex):
    spheres = sphere_coords()
    rng = np.random.default_rng(1)
    network = np.zeros(20484, bool)
    network[np.flatnonzero(cortex)[:1500]] = True
    values = np.where(network, 1.0, 0.0) + rng.normal(size=20484) * 0.01
    r = spin_test(values, cortex, {"net": network, "none": ~network}, spheres, 50, 0.05, rng)
    assert r["none"]["overlap"] == 0.0 and r["none"]["p_spin"] == 1.0
    r = r["net"]
    assert r["overlap"] == 1.0 and r["p_spin"] < 0.05
    assert overlap_fraction(values, cortex, network, 0.05) == 1.0


def test_univariate_task_means_from_shards(tmp_path):
    import importlib.util
    from tribeloc import ROOT
    spec = importlib.util.spec_from_file_location("uni", ROOT / "scripts" / "06_univariate.py")
    uni = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uni)
    table = pd.DataFrame(dict(stim_id=[f"s{i}" for i in range(8)], task=["a"] * 4 + ["b"] * 4,
                              item=[0] * 4 + [0] * 4, task_index=[0] * 4 + [1] * 4))
    x = np.arange(8.0)[:, None] * np.ones((8, 20484))
    np.savez(tmp_path / "shard_0000.npz", stim_id=table.stim_id.to_numpy()[:4].astype(str), full=x[:4])
    np.savez(tmp_path / "shard_0001.npz", stim_id=table.stim_id.to_numpy()[4:][::-1].astype(str), full=x[4:][::-1])
    m = uni.task_means(table, ["full"], tmp_path, shard_size=4)["full"]
    assert np.allclose(m[:, 0], [1.5, 5.5])
