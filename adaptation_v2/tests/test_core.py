"""Test di core.py. Lancia da adaptation_v2/:   python -m pytest -v

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


# --------------------------------------------------------------- gradiente
def test_gradient_exact_on_linear(square_mesh):
    pts, tri = square_mesh
    x, y = pts[:, 0], pts[:, 1]
    grad, _ = core.gradient_lsq(pts, 3 * x - 2 * y, core.node_neighbors(len(pts), tri))
    ok = np.isfinite(grad).all(axis=1)
    assert ok.mean() > 0.99
    np.testing.assert_allclose(grad[ok, 0], 3.0, atol=1e-8)
    np.testing.assert_allclose(grad[ok, 1], -2.0, atol=1e-8)


def test_gradient_exact_on_quadratic(square_mesh):
    pts, tri = square_mesh
    x, y = pts[:, 0], pts[:, 1]
    grad, _ = core.gradient_lsq(pts, x**2 + x * y, core.node_neighbors(len(pts), tri))
    ok = np.isfinite(grad).all(axis=1)
    np.testing.assert_allclose(grad[ok, 0], (2 * x + y)[ok], atol=1e-8)
    np.testing.assert_allclose(grad[ok, 1], x[ok], atol=1e-8)


# --------------------------------------------------------------- flussi
def test_fluxes_uniform_flow():
    n, gamma, M = 10, 1.4, 2.0
    p_inf = 1 / (gamma * M**2)
    rho, rho_u, rho_v = np.ones(n), np.ones(n), np.zeros(n)
    rho_E = p_inf / (gamma - 1) + 0.5 * rho
    Fx, Fy, p = core.euler_fluxes(rho, rho_u, rho_v, rho_E, gamma)
    assert Fx.shape == (n, 4) and Fy.shape == (n, 4)
    np.testing.assert_allclose(p, p_inf)
    np.testing.assert_allclose(Fx, np.broadcast_to([1, 1 + p_inf, 0, rho_E[0] + p_inf], (n, 4)), atol=1e-12)
    np.testing.assert_allclose(Fy, np.broadcast_to([0, 0, p_inf, 0], (n, 4)), atol=1e-12)


# --------------------------------------------------------------- goal-oriented
def test_goal_hessian_single_term(square_mesh):
    pts, tri = square_mesh
    x, y = pts[:, 0], pts[:, 1]
    n = len(pts)
    Fx, Fy = np.zeros((n, 4)), np.zeros((n, 4))
    Fx[:, 1] = 100 * x**2 + y**2              # un solo flusso non nullo
    g = np.zeros((n, 4, 2))
    g[:, 1, 0] = 1.0                          # peso: d psi_1 / dx = 1
    H_go = core.goal_oriented_hessian(pts, tri, Fx, Fy, g)
    np.testing.assert_allclose(H_go, np.broadcast_to([[200, 0], [0, 2]], (n, 2, 2)), atol=1e-6)
    # segno di psi ininfluente, scala lineare
    np.testing.assert_allclose(core.goal_oriented_hessian(pts, tri, Fx, Fy, -g), H_go)
    np.testing.assert_allclose(core.goal_oriented_hessian(pts, tri, Fx, Fy, 3 * g), 3 * H_go, rtol=1e-12)
    # peso nella direzione sbagliata (d psi/dy su un flusso in x): nessun contributo
    g2 = np.zeros((n, 4, 2))
    g2[:, 1, 1] = 1.0
    np.testing.assert_allclose(core.goal_oriented_hessian(pts, tri, Fx, Fy, g2),
                               np.broadcast_to(1e-8 * np.eye(2), (n, 2, 2)), atol=1e-12)


def test_goal_hessian_zero_adjoint(square_mesh):
    pts, tri = square_mesh
    n = len(pts)
    F = np.random.default_rng(0).normal(size=(n, 4))
    H_go = core.goal_oriented_hessian(pts, tri, F, F, np.zeros((n, 4, 2)), eigen_floor=1e-8)
    np.testing.assert_allclose(H_go, np.broadcast_to(1e-8 * np.eye(2), (n, 2, 2)))


# --------------------------------------------------------------- soglia goal-oriented
def test_goal_hessian_uniform_flow_gives_floor(square_mesh):
    """Flussi uniformi (Hessiano nullo) e aggiunto con gradiente grande:
    H_go deve valere esattamente la soglia, indipendentemente da grad_psi."""
    pts, tri = square_mesh
    n = len(pts)
    F = np.ones((n, 4))
    g = np.full((n, 4, 2), 50.0)
    H_go = core.goal_oriented_hessian(pts, tri, F, F, g, eigen_floor=1e-8)
    np.testing.assert_allclose(H_go, np.broadcast_to(1e-8 * np.eye(2), (n, 2, 2)), rtol=1e-6)


# --------------------------------------------------------------- qualita' UGAWG
def test_quality_reference_triangles():
    tri = np.array([[0, 1, 2]])
    # equilatero di lato 1, metrica identita' -> 1
    eq = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    I3 = np.array([np.eye(2)] * 3)
    assert core.element_quality(eq, tri, I3)[0] == pytest.approx(1.0)
    # rettangolo isoscele, identita' -> sqrt(3)/2
    rt = np.array([[0, 0], [1, 0], [0, 1.0]])
    assert core.element_quality(rt, tri, I3)[0] == pytest.approx(np.sqrt(3) / 2)
    # schiacciato di un fattore 10 in y, metrica diag(1, 100) -> equilatero nella metrica
    sq = np.array([[0, 0], [1, 0], [0.5, 0.1 * np.sqrt(3) / 2]])
    M = np.array([np.diag([1.0, 100.0])] * 3)
    assert core.element_quality(sq, tri, M)[0] == pytest.approx(1.0)
    # invarianza di scala
    assert core.element_quality(7 * rt, tri, I3)[0] == pytest.approx(np.sqrt(3) / 2)


def test_quality_uses_vertex_with_largest_det():
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]])
    M = np.array([(i + 1) * np.eye(2) for i in range(4)])
    # triangolo [1, 3, 2]: la metrica scelta deve essere 4 I (nodo 3), scala uniforme -> Q geometrico
    Q = core.element_quality(pts, np.array([[1, 3, 2]]), M)
    assert Q[0] == pytest.approx(np.sqrt(3) / 2)


def test_metric_edge_lengths():
    pts = np.array([[0, 0], [2, 0.0]])
    edges = np.array([[0, 1]])
    # metrica identita': lunghezza euclidea
    I2 = np.array([np.eye(2)] * 2)
    assert core.metric_edge_lengths(pts, edges, I2)[0] == pytest.approx(2.0)
    # metriche diverse ai due estremi: media logaritmica di L_a = 2 e L_b = 1
    M = np.array([np.eye(2), 0.25 * np.eye(2)])
    assert core.metric_edge_lengths(pts, edges, M)[0] == pytest.approx(1 / np.log(2))


def test_interpolate_linear_field_exactly(square_mesh):
    pts, tri = square_mesh
    rng = np.random.default_rng(3)
    new = rng.uniform(-0.99, 0.99, size=(500, 2))
    f = 2 * pts[:, 0] - 3 * pts[:, 1] + 1
    out = core.interpolate_nodal(pts, tri, f, new)
    np.testing.assert_allclose(out, 2 * new[:, 0] - 3 * new[:, 1] + 1, atol=1e-10)
    # anche per tensori (la metrica)
    M = np.einsum("n,ij->nij", 1 + pts[:, 0] ** 2, np.eye(2))
    Mi = core.interpolate_nodal(pts, tri, M, new)
    assert Mi.shape == (500, 2, 2)


def test_quality_report_on_same_mesh(square_mesh):
    pts, tri = square_mesh
    M = np.array([np.eye(2) * 400.0] * len(pts))        # taglia ~0.05
    rep = core.mesh_quality_report(pts, tri, M, pts, tri)
    assert rep["mesh_complexity"] == pytest.approx(np.sqrt(3) / 4 * len(tri), abs=0.1)
    for key in ("q_min", "q_median", "q_below_04_pct", "edges_unit_pct"):
        assert key in rep


# --------------------------------------------------------------- spigoli
def test_corner_refinement():
    pts = np.array([[1.0, 0.0], [1.001, 0.0], [3.0, 0.0]])
    # metrica anisotropa grossa: taglie 0.1 in x e 0.01 in y
    M = np.array([np.diag([1 / 0.1**2, 1 / 0.01**2])] * 3)
    M2, n = core.corner_refinement(pts, M, [(1.0, 0.0)], h_corner=1e-4, grading=0.2)
    # sullo spigolo: isotropa con taglia 1e-4
    np.testing.assert_allclose(M2[0], np.eye(2) / 1e-4**2, rtol=1e-10)
    # a distanza 1e-3: h = 1e-4 + 0.2e-3 = 3e-4
    np.testing.assert_allclose(M2[1], np.eye(2) / 3e-4**2, rtol=1e-10)
    # lontano (d = 2, h = 0.4): la metrica non cambia
    np.testing.assert_allclose(M2[2], M[2])
    assert n == 2
