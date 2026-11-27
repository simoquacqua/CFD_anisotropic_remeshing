"""Test di core.py. Lancia da adaptation/:   python -m pytest -v

Ordine consigliato: implementa le funzioni nell'ordine dei test e fai
girare pytest ogni volta (python -m pytest -v -x si ferma al primo errore).
"""
import numpy as np
import pytest

import core
from conftest import quadratic_field, shock_like_field


# --------------------------------------------------------------- utilita'
def test_triangle_areas_sum_to_domain(square_mesh):
    pts, tri = square_mesh
    assert core.triangle_areas(pts, tri).sum() == pytest.approx(4.0)


# --------------------------------------------------------------- 1. Hessiano
def test_hessian_exact_on_quadratic(square_mesh):
    pts, tri = square_mesh
    values, H_exact = quadratic_field(pts, 35.0)
    H, cond = core.hessian_lsq(pts, values, core.node_neighbors(len(pts), tri))
    ok = np.isfinite(H).all(axis=(1, 2))
    assert ok.mean() > 0.99
    assert np.abs(H[ok] - H_exact).max() < 1e-6
    assert np.all(cond[ok] >= 1.0)


def test_hessian_nan_when_stencil_too_small():
    # un solo triangolo: nessun nodo ha 5 vicini nei due anelli
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = np.array([[0, 1, 2]])
    H, _ = core.hessian_lsq(pts, np.zeros(3), core.node_neighbors(3, tri))
    assert np.isnan(H).all()


# --------------------------------------------------------------- 2. |H|
def test_abs_hessian_is_spd_and_handles_nan():
    H = np.array([[[2.0, 0.0], [0.0, -3.0]],          # indefinito
                  [[np.nan, 0.0], [0.0, 1.0]],        # non valido
                  [[0.0, 0.0], [0.0, 0.0]]])          # nullo
    A = core.abs_hessian(H, eigen_floor=1e-6)
    lam, _ = np.linalg.eigh(A)
    np.testing.assert_allclose(lam[0], [2.0, 3.0])
    np.testing.assert_allclose(lam[1], [1e-6, 1e-6])
    np.testing.assert_allclose(lam[2], [1e-6, 1e-6])
    np.testing.assert_allclose(A, A.transpose(0, 2, 1))


# --------------------------------------------------------------- 3. metrica Lp
def test_metric_lp_direction_and_anisotropy(square_mesh):
    pts, _ = square_mesh
    _, H_exact = quadratic_field(pts, 35.0)
    H = np.broadcast_to(H_exact, (len(pts), 2, 2)).copy()
    M0 = core.metric_lp_unscaled(core.abs_hessian(H), p=2)
    mu, V = np.linalg.eigh(M0)
    # rapporto delle taglie = sqrt(rapporto autovalori di H) = sqrt(100)
    np.testing.assert_allclose(np.sqrt(mu[:, 1] / mu[:, 0]), 10.0, rtol=1e-8)
    # direzione di taglia piccola (autovalore grande) = asse x'
    a = np.deg2rad(35.0)
    np.testing.assert_allclose(np.abs(V[:, :, 1] @ [np.cos(a), np.sin(a)]), 1.0)


def test_metric_lp_det_scaling():
    # det(M0) = det(|H|)^(1 - 2/(2p+2)) = det(|H|)^(p/(p+1))
    Habs = np.array([[[4.0, 0.0], [0.0, 9.0]]])
    for p in (1.0, 2.0, 4.0):
        M0 = core.metric_lp_unscaled(Habs, p=p)
        assert np.linalg.det(M0[0]) == pytest.approx(36.0 ** (p / (p + 1)))


# --------------------------------------------------------------- 4. clip
def test_clip_metric_bounds():
    rng = np.random.default_rng(1)
    lam = 10.0 ** rng.uniform(-4, 6, size=(500, 2))
    th = rng.uniform(0, np.pi, 500)
    V = np.stack([np.stack([np.cos(th), np.sin(th)], -1),
                  np.stack([-np.sin(th), np.cos(th)], -1)], axis=-1)
    M = core.from_eig(lam, V)
    Mc = core.clip_metric(M, h_min=1e-2, h_max=0.3, max_aspect_ratio=10)
    mu, Vc = np.linalg.eigh(Mc)
    h = 1 / np.sqrt(mu)
    assert h.min() >= 1e-2 * (1 - 1e-9)
    assert h.max() <= 0.3 * (1 + 1e-9)
    assert (h[:, 0] / h[:, 1]).max() <= 10 * (1 + 1e-9)
    # le direzioni principali non devono cambiare (dove restano distinte)
    distinct = mu[:, 1] > 1.01 * mu[:, 0]
    align = np.abs(np.einsum("ni,ni->n", Vc[:, :, 1], np.linalg.eigh(M)[1][:, :, 1]))
    np.testing.assert_allclose(align[distinct], 1.0, atol=1e-6)


def test_clip_metric_none_is_identity():
    M = np.array([[[1e-9, 0.0], [0.0, 1e9]]])
    np.testing.assert_allclose(core.clip_metric(M), M)


# --------------------------------------------------------------- 5. complessita'
def test_complexity_of_uniform_metric(square_mesh):
    pts, tri = square_mesh
    h = 0.1
    M = np.broadcast_to(np.eye(2) / h**2, (len(pts), 2, 2))
    # sqrt(det M) = 1/h^2 costante, dominio di area 4
    assert core.metric_complexity(M, pts, tri) == pytest.approx(4 / h**2)


def test_complexity_scales_linearly(square_mesh):
    pts, tri = square_mesh
    values, _ = quadratic_field(pts)
    H, _ = core.hessian_lsq(pts, values, core.node_neighbors(len(pts), tri))
    M0 = core.metric_lp_unscaled(core.abs_hessian(H))
    c1 = core.metric_complexity(M0, pts, tri)
    assert core.metric_complexity(7.0 * M0, pts, tri) == pytest.approx(7.0 * c1)


def test_normalize_without_clip(square_mesh):
    pts, tri = square_mesh
    values, _ = quadratic_field(pts)
    H, _ = core.hessian_lsq(pts, values, core.node_neighbors(len(pts), tri))
    M0 = core.metric_lp_unscaled(core.abs_hessian(H))
    M, D = core.normalize_metric(M0, pts, tri, target=2000)
    assert core.metric_complexity(M, pts, tri) == pytest.approx(2000, rel=1e-4)
    np.testing.assert_allclose(M, D * M0)


def test_normalize_hits_target_with_clip(square_mesh):
    """Il test chiave: caso tipo urto, dove il clip con h_max e' attivo."""
    pts, tri = square_mesh
    values = shock_like_field(pts)
    H, _ = core.hessian_lsq(pts, values, core.node_neighbors(len(pts), tri))
    M0 = core.metric_lp_unscaled(core.abs_hessian(H))
    kw = dict(h_min=1e-3, h_max=0.1, max_aspect_ratio=10)
    target = 1000

    # approccio attuale (D scelto prima del clip): sbaglia la complessita'
    D_naive = target / core.metric_complexity(M0, pts, tri)
    c_naive = core.metric_complexity(core.clip_metric(D_naive * M0, **kw), pts, tri)
    assert abs(c_naive / target - 1) > 0.05, "il caso test dovrebbe mostrare il problema"

    # approccio corretto
    M, D = core.normalize_metric(M0, pts, tri, target, **kw)
    assert core.metric_complexity(M, pts, tri) == pytest.approx(target, rel=1e-3)
    np.testing.assert_allclose(M, core.clip_metric(D * M0, **kw))


def test_normalize_unreachable_target_raises(square_mesh):
    pts, tri = square_mesh
    M0 = np.broadcast_to(np.eye(2), (len(pts), 2, 2))
    # con h_max = 0.1 la complessita' minima e' 4 / 0.01 = 400
    with pytest.raises(ValueError):
        core.normalize_metric(M0, pts, tri, target=100, h_max=0.1)
