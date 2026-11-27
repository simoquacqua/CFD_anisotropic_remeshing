import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


GAMMA = 1.4


def fmt(values, per_line=10):
    values = np.asarray(values, dtype=float)
    return ",\n  ".join(
        ", ".join(f"{v:.10g}" for v in values[i:i + per_line])
        for i in range(0, len(values), per_line)
    )


def createGeoFile(path, x, y):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"sx[] = {{ {fmt(x)} }};\n")
        f.write(f"sy[] = {{ {fmt(y)} }};\n")


def mesh_connectivity(mesh):
    """Return node adjacency and cell vertex ids for a 2D triangular mesh."""
    if mesh.n_cells == 0:
        raise ValueError("Empty mesh")
    neighbors = [set() for _ in range(mesh.n_points)]
    triangles = []
    for k in range(mesh.n_cells):
        cell = mesh.get_cell(k)
        ids = np.asarray(cell.point_ids, dtype=int)
        if len(ids) != 3 or cell.type != pv.CellType.TRIANGLE:
            raise ValueError("Only 2D triangle cells are supported")
        triangles.append(ids)
        for i in ids:
            neighbors[i].update(j for j in ids if j != i)
    return neighbors, np.asarray(triangles, dtype=int)


def compute_hessian_lsq(mesh, values, neighbors=None, max_condition=1e10):
    """Two-ring quadratic least-squares nodal Hessian on a 2D mesh."""
    xy = np.asarray(mesh.points[:, :2], dtype=float)
    n = mesh.n_points
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite nodal scalar array")
    if neighbors is None:
        neighbors, _ = mesh_connectivity(mesh)

    H = np.full((n, 2, 2), np.nan, dtype=float)
    condition = np.full(n, np.nan, dtype=float)

    for i in range(n):
        stencil = set(neighbors[i])
        for j in neighbors[i]:
            stencil.update(neighbors[j])
        stencil.discard(i)
        if len(stencil) < 5:
            continue

        ids = np.fromiter(sorted(stencil), dtype=int)
        delta = xy[ids] - xy[i]
        scale = np.max(np.linalg.norm(delta, axis=1))
        if not np.isfinite(scale) or scale <= 0:
            continue

        dx = delta[:, 0] / scale
        dy = delta[:, 1] / scale
        A = np.column_stack((dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2))
        dp = values[ids] - values[i]
        beta, _, rank, s = np.linalg.lstsq(A, dp, rcond=None)
        if rank < 5 or s[-1] <= 0:
            continue
        condition[i] = s[0] / s[-1]
        if condition[i] > max_condition:
            continue

        Hxx, Hxy, Hyy = beta[2:] / scale**2
        H[i] = ((Hxx, Hxy), (Hxy, Hyy))
    return H, condition


def computeAnisotropicMetricField(mesh, H, TargetComplexity, p=2,
                                  EigenFloor=1e-8,
                                  h_min=None, h_max=None,
                                  max_aspect_ratio=None):
    """2D Lp interpolation-error metric; complexity is integral sqrt(det(M)) dOmega.

    Invalid Hessians are replaced by zero curvature before regularization.
    h_min/h_max and max_aspect_ratio are optional constraints; clipping them
    may change the achieved complexity.
    """
    n = mesh.n_points
    H = np.asarray(H, dtype=float)
    if H.shape != (n, 2, 2):
        raise ValueError("H must have shape (n_points, 2, 2)")
    if p < 1 or TargetComplexity <= 0 or EigenFloor <= 0:
        raise ValueError("Invalid metric parameters")
    if h_min is not None and h_min <= 0:
        raise ValueError("h_min must be positive")
    if h_max is not None and h_max <= 0:
        raise ValueError("h_max must be positive")
    if h_min is not None and h_max is not None and h_min > h_max:
        raise ValueError("h_min must be <= h_max")
    if max_aspect_ratio is not None and max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio must be >= 1")

    valid = np.isfinite(H).all(axis=(1, 2))
    if not np.any(valid):
        raise ValueError("All Hessian reconstructions failed")
    if not np.all(valid):
        print(
            f"Warning: replacing {np.count_nonzero(~valid)} invalid Hessians with zero")
    H = H.copy()
    H[~valid] = 0.0
    H = 0.5 * (H + H.transpose(0, 2, 1))
    eigenvalues, R = np.linalg.eigh(H)
    lam = np.maximum(np.abs(eigenvalues), EigenFloor)
    Habs = (R * lam[:, None, :]) @ R.transpose(0, 2, 1)
    detH = lam[:, 0] * lam[:, 1]

    areas = np.asarray(mesh.compute_cell_sizes(length=False, area=True,
                                               volume=False).cell_data["Area"])
    _, tri = mesh_connectivity(mesh)
    if np.any(areas <= 0):
        raise ValueError("Non-positive cell area")

    exponent = p / (2 * p + 2)
    rho = detH**exponent
    integral = np.sum(areas * rho[tri].mean(axis=1))
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("Invalid density integral")

    D = TargetComplexity / integral
    M = D * detH[:, None, None]**(-1 / (2 * p + 2)) * Habs

    if h_min is not None or h_max is not None or max_aspect_ratio is not None:
        m, V = np.linalg.eigh(M)
        lower = 0.0 if h_max is None else 1.0 / h_max**2
        upper = np.inf if h_min is None else 1.0 / h_min**2
        m = np.clip(m, lower, upper)
        if max_aspect_ratio is not None:
            m[:, 0] = np.maximum(m[:, 0], m[:, 1] / max_aspect_ratio**2)
        M = (V * m[:, None, :]) @ V.transpose(0, 2, 1)

    complexity = np.sum(areas * np.sqrt(np.linalg.det(M))[tri].mean(axis=1))
    print(f"Target metric complexity: {TargetComplexity:.6g}")
    print(f"Actual metric complexity: {complexity:.6g}")
    return M


def connected_components(mask, neighbors):
    """Connected components of selected nodes, using mesh edges."""
    remaining = set(np.flatnonzero(mask))
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            for j in neighbors[stack.pop()]:
                if j in remaining:
                    remaining.remove(j)
                    stack.append(j)
                    comp.append(j)
        components.append(np.asarray(comp, dtype=int))
    return sorted(components, key=len, reverse=True)


def fit_line_tls(points, samples=20):
    """Orthogonal total-least-squares segment bounded by observed candidates."""
    if len(points) < 3:
        raise ValueError("At least three points are required to fit a front")
    centre = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - centre, full_matrices=False)
    if singular[0] <= 0:
        raise ValueError("Degenerate line fit")
    direction = vt[0]
    t = (points - centre) @ direction
    a, b = np.quantile(t, [0.02, 0.98])
    return centre + np.linspace(a, b, samples)[:, None] * direction


def shockDetection(VTUfile, field="Pressure", grad_threshold=0.1,
                   ducros_threshold=0.7, min_component_nodes=3,
                   samples_per_front=20, diagnostic_file=None,
                   metric_file=None, target_complexity=None,
                   h_min=None, h_max=None, max_aspect_ratio=None):
    mesh = pv.read(VTUfile)
    if field not in mesh.point_data:
        raise KeyError(
            f"{field} must be in mesh.point_data; available: {list(mesh.point_data.keys())}")
    for name in ("Pressure", "Density", "Velocity"):
        if name not in mesh.point_data:
            raise KeyError(f"Missing nodal field: {name}")
    neighbors, _ = mesh_connectivity(mesh)

    xy = np.asarray(mesh.points[:, :2], dtype=float)
    pressure = np.asarray(mesh.point_data["Pressure"], dtype=float)
    density = np.asarray(mesh.point_data["Density"], dtype=float)
    velocity = np.asarray(mesh.point_data["Velocity"], dtype=float)[:, :2]
    if np.any(pressure <= 0) or np.any(density <= 0):
        raise ValueError("Pressure and Density must be positive")

    g = mesh.compute_derivative(scalars=field, gradient=True,
                                preference="point")
    grad = np.asarray(g.point_data["gradient"], dtype=float)[:, :2]
    grad_mag = np.linalg.norm(grad, axis=1)
    tiny = 1e-12
    normal = grad / np.maximum(grad_mag[:, None], tiny)
    a = np.sqrt(GAMMA * pressure / density)
    Mn = np.abs(np.einsum("ij,ij->i", velocity, normal)) / a

    derivatives = mesh.compute_derivative(scalars="Velocity", divergence=True,
                                          vorticity=True, preference="point")
    div_u = np.asarray(derivatives.point_data["divergence"], dtype=float)
    vort = np.asarray(derivatives.point_data["vorticity"], dtype=float)
    omega2 = np.sum(vort**2, axis=1) if vort.ndim == 2 else vort**2
    ducros = div_u**2 / (div_u**2 + omega2 + tiny)

    # Normalize separately in the upper and lower half-plane so one strong
    # shock does not hide a weaker shock on the other side.
    grad_relative = np.zeros_like(grad_mag)
    for half in (xy[:, 1] >= 0, xy[:, 1] < 0):
        if np.any(half):
            grad_relative[half] = grad_mag[half] / \
                max(grad_mag[half].max(), tiny)

    candidate = ((grad_relative > grad_threshold) &
                 (ducros > ducros_threshold) & (div_u < 0))

    if diagnostic_file is not None:
        mesh.point_data["shock_candidate"] = candidate.astype(np.uint8)
        mesh.point_data["grad_relative"] = grad_relative
        mesh.point_data["ducros_sensor"] = ducros
        mesh.point_data["velocity_divergence"] = div_u
        mesh.point_data["normal_mach_estimate"] = Mn
        mesh.save(diagnostic_file)

    segments = []
    for label, half in (("upper", xy[:, 1] >= 0), ("lower", xy[:, 1] < 0)):
        comps = connected_components(candidate & half, neighbors)
        comps = [c for c in comps if len(c) >= min_component_nodes]
        if not comps:
            raise RuntimeError(
                f"No sufficiently connected {label} shock candidate found. "
                "Inspect diagnostic VTU or tune sensor thresholds."
            )
        # First prototype: one dominant shock per half-plane.
        best = comps[0]
        print(
            f"{label}: selected {len(best)} nodes; {len(comps)} components passed minimum size")
        segments.append(fit_line_tls(xy[best], samples=samples_per_front))

    # Concatenation ordered from highest y to lowest y, preserving the legacy
    # sx[] and sy[] output expected by the .geo file.
    line = np.vstack(segments)
    line = line[np.argsort(line[:, 1])[::-1]]

    if target_complexity is not None:
        H, cond = compute_hessian_lsq(
            mesh, np.asarray(mesh.point_data[field]), neighbors)
        M = computeAnisotropicMetricField(mesh, H, target_complexity,
                                          h_min=h_min, h_max=h_max,
                                          max_aspect_ratio=max_aspect_ratio)
        if metric_file is None:
            metric_file = str(Path(VTUfile).with_suffix("")) + "_metric.npz"
        # This file stores the metric for later use; Gmsh does NOT automatically
        # read this .npz format.
        np.savez(metric_file, points=xy, metric=M, hessian=H, condition=cond)
        print(f"Metric saved to {metric_file} (not yet imported into Gmsh)")

    return line[:, 0], line[:, 1]


def main():
    parser = argparse.ArgumentParser(
        description="Detect steady oblique shocks and write Gmsh point arrays")
    parser.add_argument("-VTUfile", required=True, type=Path)
    parser.add_argument("-GEOfile", type=Path,
                        default=Path.cwd() / "shockPoints.geo")
    parser.add_argument("-field", default="Pressure")
    parser.add_argument("--grad-threshold", type=float, default=0.1)
    parser.add_argument("--ducros-threshold", type=float, default=0.7)
    parser.add_argument("--min-component-nodes", type=int, default=3)
    parser.add_argument("--diagnostic-vtu", type=Path, default=None)
    parser.add_argument("--target-complexity", type=float, default=None)
    parser.add_argument("--metric-file", type=Path, default=None)
    parser.add_argument("--h-min", type=float, default=None)
    parser.add_argument("--h-max", type=float, default=None)
    parser.add_argument("--max-aspect-ratio", type=float, default=None)
    args = parser.parse_args()

    x, y = shockDetection(
        args.VTUfile, field=args.field,
        grad_threshold=args.grad_threshold,
        ducros_threshold=args.ducros_threshold,
        min_component_nodes=args.min_component_nodes,
        diagnostic_file=args.diagnostic_vtu,
        metric_file=args.metric_file,
        target_complexity=args.target_complexity,
        h_min=args.h_min, h_max=args.h_max,
        max_aspect_ratio=args.max_aspect_ratio,
    )
    createGeoFile(args.GEOfile, x, y)
    print(f"Wrote {len(x)} shock points to {args.GEOfile}")


if __name__ == "__main__":
    main()
