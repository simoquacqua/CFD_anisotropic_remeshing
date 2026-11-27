
#!/usr/bin/env python3
"""
Pressure Hessian reconstruction using local quadratic
least squares (LSQ) on an unstructured triangular mesh.

Outputs:
  - pressure_hessian.vtu
  - hessian_validation.vtu

The Hessian is reconstructed directly from nodal values.
The pressure gradient is also computed with PyVista for
an independent directional comparison.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# 1. LOCAL QUADRATIC LEAST-SQUARES HESSIAN
# ============================================================

def compute_hessian_lsq(mesh, values, max_condition=1e10):
    """
    Reconstruct a 2D nodal Hessian with quadratic LSQ.

    Uses two rings of neighboring nodes.

    Returns
    -------
    H : ndarray, shape (n_points, 2, 2)
        Reconstructed Hessian.
    condition : ndarray, shape (n_points,)
        Condition number of the local LSQ matrix.
    """

    coords = mesh.points[:, :2]
    n_points = mesh.n_points

    values = np.asarray(values, dtype=float)

    if values.shape != (n_points,):
        raise ValueError("Values must be one scalar per node")

    # Build node connectivity
    neighbors = [set() for _ in range(n_points)]

    for cell_id in range(mesh.n_cells):
        ids = list(mesh.get_cell(cell_id).point_ids)

        for i in ids:
            neighbors[i].update(j for j in ids if j != i)

    # Allocate arrays
    H = np.full((n_points, 2, 2), np.nan)
    condition = np.full(n_points, np.nan)

    for i in range(n_points):

        # First ring
        ring1 = neighbors[i]

        # Second ring
        stencil = set(ring1)

        for j in ring1:
            stencil.update(neighbors[j])

        stencil.discard(i)

        ids = np.array(sorted(stencil), dtype=int)

        if len(ids) < 5:
            continue

        # Relative coordinates and scalar differences
        delta = coords[ids] - coords[i]
        dp = values[ids] - values[i]

        # Scale local coordinates
        scale = np.max(np.linalg.norm(delta, axis=1))

        if not np.isfinite(scale) or scale <= 0:
            continue

        dx = delta[:, 0] / scale
        dy = delta[:, 1] / scale

        # Quadratic Taylor reconstruction matrix
        A = np.column_stack([
            dx,
            dy,
            0.5 * dx**2,
            dx * dy,
            0.5 * dy**2
        ])

        # Solve least-squares problem
        beta, residuals, rank, s = np.linalg.lstsq(
            A, dp, rcond=None
        )

        if rank < 5:
            continue

        # Check numerical conditioning
        if s[-1] <= 0:
            continue

        cond = s[0] / s[-1]
        condition[i] = cond

        if cond > max_condition:
            continue

        # Recover physical second derivatives
        Hxx = beta[2] / scale**2
        Hxy = beta[3] / scale**2
        Hyy = beta[4] / scale**2

        H[i] = [
            [Hxx, Hxy],
            [Hxy, Hyy]
        ]

    return H, condition


# ============================================================
# 2. ANALYTICAL VALIDATION
# ============================================================

def validate_hessian(mesh, output_path):

    coords = mesh.points
    x = coords[:, 0]
    y = coords[:, 1]

    # Analytical quadratic field
    p_test = 100.0 * x**2 + y**2

    # Numerical Hessian
    H_test, condition = compute_hessian_lsq(
        mesh, p_test
    )

    valid = np.isfinite(H_test).all(axis=(1, 2))

    if not np.any(valid):
        raise ValueError("No valid Hessians in validation")

    Hxx = H_test[:, 0, 0]
    Hxy = H_test[:, 0, 1]
    Hyy = H_test[:, 1, 1]

    # Exact Hessian: [[200, 0], [0, 2]]
    err_xx = np.abs(Hxx - 200.0)
    err_xy = np.abs(Hxy)
    err_yy = np.abs(Hyy - 2.0)

    print("\n========== HESSIAN LSQ VALIDATION ==========")
    print("Valid Hessians:", np.count_nonzero(valid))
    print("Total nodes:", mesh.n_points)

    print("\nExpected: Hxx=200, Hxy=0, Hyy=2")

    print("\nMean numerical values:")
    print("Hxx:", np.mean(Hxx[valid]))
    print("Hxy:", np.mean(Hxy[valid]))
    print("Hyy:", np.mean(Hyy[valid]))

    print("\nMean absolute errors:")
    print("Hxx:", np.mean(err_xx[valid]))
    print("Hxy:", np.mean(err_xy[valid]))
    print("Hyy:", np.mean(err_yy[valid]))

    print("\nMaximum absolute errors:")
    print("Hxx:", np.max(err_xx[valid]))
    print("Hxy:", np.max(err_xy[valid]))
    print("Hyy:", np.max(err_yy[valid]))

    # Save analytical test independently
    test_mesh = mesh.copy(deep=True)

    test_mesh.point_data["PressureTest"] = p_test
    test_mesh.point_data["HessianTestXX"] = Hxx
    test_mesh.point_data["HessianTestXY"] = Hxy
    test_mesh.point_data["HessianTestYY"] = Hyy

    test_mesh.point_data["HessianTestErrorXX"] = err_xx
    test_mesh.point_data["HessianTestErrorXY"] = err_xy
    test_mesh.point_data["HessianTestErrorYY"] = err_yy

    test_mesh.point_data["ValidHessian"] = valid.astype(
        np.uint8
    )
    test_mesh.point_data["LSQCondition"] = condition

    test_mesh.save(str(output_path))


# ============================================================
# 3. COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Compute a nodal Hessian using quadratic LSQ"
)

parser.add_argument(
    "--input",
    type=str,
    default="../Baseline/Results/flow_00131.vtu",
    help="Input SU2 VTU file"
)

parser.add_argument(
    "--output",
    type=str,
    default="../Diagnostics/pressure_hessian.vtu",
    help="Output diagnostic VTU file"
)

parser.add_argument(
    "--field",
    type=str,
    default="Pressure",
    help="Scalar field used to compute the Hessian"
)

args = parser.parse_args()


# ============================================================
# 4. READ MESH AND SCALAR FIELD
# ============================================================

mesh = pv.read(args.input)

if args.field not in mesh.point_data:
    raise ValueError(
        f"Field '{args.field}' not found in PointData. "
        f"Available: {list(mesh.point_data.keys())}"
    )

values = np.asarray(
    mesh.point_data[args.field],
    dtype=float
)

if values.ndim != 1:
    raise ValueError("Selected field must be scalar")

n_points = mesh.n_points

print("\n========== INPUT MESH ==========")
print("Points:", n_points)
print("Cells:", mesh.n_cells)
print("Field:", args.field)


# ============================================================
# 5. COMPUTE HESSIAN WITH LSQ
# ============================================================

H, condition = compute_hessian_lsq(mesh, values)

valid_hessian = np.isfinite(H).all(axis=(1, 2))

print("\n========== LSQ RECONSTRUCTION ==========")
print("Valid Hessians:", np.count_nonzero(valid_hessian))
print("Invalid Hessians:", np.count_nonzero(~valid_hessian))

if not np.any(valid_hessian):
    raise ValueError("No valid Hessians were reconstructed")

# Save individual tensor components
mesh.point_data["HessianXX"] = H[:, 0, 0]
mesh.point_data["HessianXY"] = H[:, 0, 1]
mesh.point_data["HessianYY"] = H[:, 1, 1]

mesh.point_data["ValidHessian"] = (
    valid_hessian.astype(np.uint8)
)

mesh.point_data["LSQCondition"] = condition


# ============================================================
# 6. EIGENVALUES AND EIGENVECTORS
# ============================================================

# Allocate full-size arrays.
# Invalid Hessians remain NaN, not artificial zero tensors.

lambda_max = np.full(n_points, np.nan)
lambda_min = np.full(n_points, np.nan)

v_max = np.zeros((n_points, 2))
anisotropy_indicator = np.full(n_points, np.nan)

valid_ids = np.flatnonzero(valid_hessian)

# Eigendecomposition only on valid Hessians
eigenvalues, eigenvectors = np.linalg.eigh(
    H[valid_hessian]
)

abs_eigenvalues = np.abs(eigenvalues)

id_max = np.argmax(abs_eigenvalues, axis=1)
id_min = 1 - id_max

rows = np.arange(len(valid_ids))

lambda_max[valid_ids] = abs_eigenvalues[
    rows, id_max
]

lambda_min[valid_ids] = abs_eigenvalues[
    rows, id_min
]

v_max[valid_ids] = eigenvectors[
    rows, :, id_max
]

# Relative separation between principal curvatures
denom = lambda_max + lambda_min

nonzero_curvature = (
    valid_hessian
    & np.isfinite(denom)
    & (denom > 0)
)

anisotropy_indicator[nonzero_curvature] = (
    (lambda_max[nonzero_curvature]
     - lambda_min[nonzero_curvature])
    / denom[nonzero_curvature]
)

mesh.point_data["LambdaMaxAbs"] = lambda_max
mesh.point_data["LambdaMinAbs"] = lambda_min

mesh.point_data["HessianAnisotropyIndicator"] = (
    anisotropy_indicator
)


# ============================================================
# 7. PRESSURE GRADIENT (INDEPENDENT COMPARISON)
# ============================================================

# Compute the gradient with PyVista.
# It is used for directional comparison, not to
# construct the LSQ Hessian.

g = mesh.compute_derivative(
    scalars=args.field,
    gradient=True,
    preference="point"
)

grad = np.asarray(g.point_data["gradient"])

grad_2d = grad[:, :2]

grad_norm = np.linalg.norm(
    grad_2d,
    axis=1
)

valid_gradient = (
    np.isfinite(grad_norm)
    & (grad_norm > 1e-12)
)

n_grad = np.zeros((n_points, 2))

n_grad[valid_gradient] = (
    grad_2d[valid_gradient]
    / grad_norm[valid_gradient, None]
)

mesh.point_data["GradNorm"] = grad_norm
mesh.point_data["ValidGradient"] = (
    valid_gradient.astype(np.uint8)
)


# ============================================================
# 8. GRADIENT-HESSIAN DIRECTIONAL ALIGNMENT
# ============================================================

valid_alignment = (
    valid_gradient
    & valid_hessian
    & nonzero_curvature
)

alignment = np.full(n_points, np.nan)

alignment[valid_alignment] = np.abs(
    np.sum(
        n_grad[valid_alignment]
        * v_max[valid_alignment],
        axis=1
    )
)

alignment[valid_alignment] = np.clip(
    alignment[valid_alignment], 0.0, 1.0
)

mesh.point_data["HessianAlignment"] = alignment

mesh.point_data["ValidAlignment"] = (
    valid_alignment.astype(np.uint8)
)

print("\n========== DIRECTIONAL ALIGNMENT ==========")

print("Valid gradient nodes:",
      np.count_nonzero(valid_gradient))

print("Valid alignment nodes:",
      np.count_nonzero(valid_alignment))

if np.any(valid_alignment):
    print("Alignment min:",
          np.nanmin(alignment))
    print("Alignment max:",
          np.nanmax(alignment))
    print("Alignment mean:",
          np.nanmean(alignment))
    print("Alignment median:",
          np.nanmedian(alignment))


# ============================================================
# 8B. HESSIAN RELIABILITY CHECK
# ============================================================

gradient_threshold = 0.01 * np.nanmax(grad_norm)
curvature_threshold = 0.01 * np.nanmax(lambda_max)

reliable = (
    valid_alignment
    & (grad_norm > gradient_threshold)
    & (lambda_max > curvature_threshold)
    & (anisotropy_indicator > 0.5)
)

mesh.point_data["ReliableHessian"] = reliable.astype(np.uint8)

print("\n========== HESSIAN RELIABILITY ==========")
print("Gradient threshold:", gradient_threshold)
print("Curvature threshold:", curvature_threshold)
print("Reliable nodes:", np.count_nonzero(reliable))
print("Total nodes:", mesh.n_points)


# ============================================================
# 9. EXPORT PRINCIPAL DIRECTIONS TO PARAVIEW
# ============================================================

# ParaView uses three-component vectors conveniently.
direction_3d = np.zeros((n_points, 3))

direction_3d[:, :2] = v_max

mesh.point_data["HessianDirection"] = direction_3d


# ============================================================
# 10. SAVE CFD DIAGNOSTIC
# ============================================================

output_path = Path(args.output)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

mesh.save(str(output_path))

print("\nCFD diagnostic saved to:", output_path)


# ============================================================
# 11. ANALYTICAL VALIDATION
# ============================================================

validation_path = (
    output_path.parent / "hessian_validation.vtu"
)

validate_hessian(mesh, validation_path)

print("\nValidation saved to:", validation_path)
