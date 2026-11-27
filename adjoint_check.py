"""Aggiunto riscalato e indicatore d'errore goal, da aprire in ParaView.

L'aggiunto scritto da SU2 (adjoint.vtu) e' U_bar = P^T lambda, con
|V_i|/dt_i = s_i/CFL proporzionale alla taglia della cella: sulle celle grandi
U_bar e' grande anche dove lambda non lo e'. Questo script scrive:

    lambda_rho, ..., lambda_E   lambda_i = CFL * U_bar_i / s_i   (aggiunto "puntuale")
    grad_lambda_rho             |grad lambda_rho|
    Ubar_rho, grad_Ubar_rho     i valori grezzi di SU2, per confronto
    error_indicator             traccia di H_go = sum_j |d lambda_j/dx||H(F_j)| + |d lambda_j/dy||H(G_j)|
                                (e' quello che guida la metrica goal: serve sia l'aggiunto sia la soluzione)
    s                           somma dei raggi spettrali sulle facce

Uso (dalla cartella test-simone):
    python3 adjoint_check.py runs/<caso>/<run>/iter_04 --cfl 5
Scrive iter_04/Results/adjoint_check.vtu. La CFL e' quella dell'aggiunto (adjoint_cfl nel log
del ciclo): cambia solo la scala, non la forma dei campi.
"""
import argparse
from pathlib import Path

import numpy as np

from adaptation_v2 import core, files


def write_vtu(path, points, triangles, fields):
    """VTU (XML, ASCII): griglia non strutturata di triangoli con campi nodali.
    I valori non finiti (nan, inf) vengono sostituiti con 0, altrimenti ParaView non legge il file."""
    n, m = len(points), len(triangles)
    xyz = np.zeros((n, 3))
    xyz[:, :2] = np.asarray(points)[:, :2]
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(
            f'<UnstructuredGrid>\n<Piece NumberOfPoints="{n}" NumberOfCells="{m}">\n')
        f.write('<PointData>\n')
        for name, values in fields.items():
            v = np.asarray(values, dtype=float)
            bad = ~np.isfinite(v)
            if bad.any():
                print(
                    f"  attenzione: {bad.sum()} valori non finiti in {name}, messi a 0")
                v = np.where(bad, 0.0, v)
            f.write(
                f'<DataArray type="Float64" Name="{name}" format="ascii">\n')
            f.write(" ".join(f"{x:.8e}" for x in v) + "\n</DataArray>\n")
        f.write(
            '</PointData>\n<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(" ".join(f"{x:.10e}" for x in xyz.ravel()) +
                "\n</DataArray>\n</Points>\n<Cells>\n")
        f.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(" ".join(str(int(k))
                for k in np.asarray(triangles).ravel()) + "\n</DataArray>\n")
        f.write('<DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(" ".join(str(3 * (k + 1))
                for k in range(m)) + "\n</DataArray>\n")
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write(" ".join(
            ["5"] * m) + "\n</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("iter_dir", type=Path,
                    help="cartella dell'iterazione (contiene Results/)")
parser.add_argument("--cfl", type=float, default=5.0,
                    help="CFL dell'aggiunto (default 5)")
args = parser.parse_args()

results = args.iter_dir / "Results"
flow = files.read_vtu(results / "flow.vtu")
adjoint = files.read_vtu(results / "adjoint.vtu")
points, triangles = flow["points"], flow["triangles"]

rho = files.get_field(flow, "Density")
rho_u = files.get_field(flow, "Momentum_x")
rho_v = files.get_field(flow, "Momentum_y")
rho_E = files.get_field(flow, "Energy")
names = ["Adjoint_Density", "Adjoint_Momentum_x",
         "Adjoint_Momentum_y", "Adjoint_Energy"]
u_bar = [files.get_field(adjoint, name) for name in names]

s = core.spectral_radius_sum(points, triangles, rho, rho_u, rho_v, rho_E)
lam = [args.cfl * ub / s for ub in u_bar]

neighbors = core.node_neighbors(len(points), triangles)
grad_lam = np.stack([core.gradient_lsq(points, l, neighbors)[0]
                    for l in lam], axis=1)   # (n, 4, 2)
grad_ubar_rho = core.gradient_lsq(points, u_bar[0], neighbors)[0]

Fx, Fy, _ = core.euler_fluxes(rho, rho_u, rho_v, rho_E)
H_go = core.goal_oriented_hessian(points, triangles, Fx, Fy, grad_lam)
indicator = np.trace(H_go, axis1=1, axis2=2)

out = results / "adjoint_check.vtu"
write_vtu(out, points, triangles, {
    "lambda_rho": lam[0], "lambda_rho_u": lam[1], "lambda_rho_v": lam[2], "lambda_E": lam[3],
    "grad_lambda_rho": np.linalg.norm(grad_lam[:, 0, :], axis=1),
    "Ubar_rho": u_bar[0],
    "grad_Ubar_rho": np.linalg.norm(grad_ubar_rho, axis=1),
    "error_indicator": indicator,
    "s": s,
})
print(f"scritto {out}")
print(f"  lambda_rho: min {lam[0].min():.3g}, max {lam[0].max():.3g}")
print(
    f"  error_indicator: max {indicator.max():.3g}, mediana {np.median(indicator):.3g}")
