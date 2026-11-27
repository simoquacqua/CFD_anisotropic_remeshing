"""
cd_breakdown.py - Da dove viene l'errore sul C_D, iterazione per iterazione.

Diamante t/c = 0.07, Euler, M = 2, alpha = 0.

Per ogni iter_XX del run:
  1. legge Results/surface_flow.csv (nodi del profilo e variabili conservative);
  2. ricava la pressione p = (gamma-1) (rhoE - |rho u|^2 / (2 rho));
  3. integra la forza lungo dorso e ventre (trapezi) -> C_D (deve coincidere con SU2);
  4. confronta ogni segmento con la pressione esatta della teoria shock-expansion
     (costante su ogni faccia) e somma l'errore per zone:
        LE (x < 0.05), facce anteriori, spigolo (0.45 < x < 0.55),
        facce posteriori, TE (x > 0.95).
  5. dalla mesh: nodi entro r = 0.01 da TE, LE e spigolo superiore.

Uso (dalla cartella test-simone):
    python3 cd_breakdown.py runs/diamond_euler_cfdladapt/NOME_RUN
"""
import sys
from pathlib import Path

import numpy as np

GAMMA = 1.4
P_INF = 1 / (GAMMA * 2.0**2)          # 0.178571 (variabili adimensionali, V = rho = 1)
Q_INF = 0.5                           # pressione dinamica
P_FRONT = 0.2226818591327492          # dopo l'urto obliquo (teoria esatta)
P_REAR = 0.1416758781647842           # dopo l'espansione di Prandtl-Meyer
CD_REF = 0.011341

ZONES = [("LE", 0.0, 0.05), ("ant", 0.05, 0.45), ("spig", 0.45, 0.55),
         ("post", 0.55, 0.95), ("TE", 0.95, 1.01)]


def surface_pressure(csv_file):
    d = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
    x, y = d[:, 1], d[:, 2]
    rho, mx, my, E = d[:, 3], d[:, 4], d[:, 5], d[:, 6]
    p = (GAMMA - 1) * (E - 0.5 * (mx**2 + my**2) / rho)
    return x, y, p


def cd_by_zone(x, y, p):
    """C_D numerico e errore per zona (contributo numerico - contributo esatto)."""
    cd = 0.0
    err = {name: 0.0 for name, _, _ in ZONES}
    for side in (+1, -1):                                 # dorso, ventre
        on_side = (np.sign(y) == side) | (np.abs(y) < 1e-12)
        idx = np.where(on_side)[0]
        idx = idx[np.argsort(x[idx])]                     # da LE a TE
        X, Y, P = x[idx], y[idx], p[idx]
        for k in range(len(X) - 1):
            dy = Y[k + 1] - Y[k]
            nx_L = -dy * side                             # componente x della normale esterna * lunghezza
            xm = 0.5 * (X[k] + X[k + 1])
            p_num = 0.5 * (P[k] + P[k + 1])
            p_ex = P_FRONT if xm < 0.5 else P_REAR
            c_num = -(p_num - P_INF) * nx_L / Q_INF
            c_ex = -(p_ex - P_INF) * nx_L / Q_INF
            cd += c_num
            for name, a, b in ZONES:
                if a <= xm < b:
                    err[name] += c_num - c_ex
    return cd, err


def nodes_near_corners(mesh_file, r=0.01):
    """Nodi entro r da TE, LE e spigolo superiore (legge solo NPOIN della mesh .su2)."""
    lines = Path(mesh_file).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("NPOIN="):
            n = int(line.split("=")[1].split()[0])
            pts = np.array([[float(v) for v in row.split()[:2]] for row in lines[i + 1:i + 1 + n]])
            break
    else:
        return None
    out = {}
    for name, c in (("TE", (1.0, 0.0)), ("LE", (0.0, 0.0)), ("spig", (0.5, 0.035))):
        out[name] = int((np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < r).sum())
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    print(f"Run: {run_dir}")
    print("errori in % del C_D di riferimento (0.011341); nodi = entro r = 0.01 dallo spigolo")
    print(f"{'iter':>7} {'C_D':>9} {'err':>7} |" + "".join(f"{n:>7}" for n, _, _ in ZONES)
          + " | nodi TE   LE spig")
    for it in sorted(run_dir.glob("iter_*")):
        csv_file = it / "Results" / "surface_flow.csv"
        if not csv_file.is_file():
            continue
        cd, err = cd_by_zone(*surface_pressure(csv_file))
        near = nodes_near_corners(it / "mesh.su2")
        line = f"{it.name:>7} {cd:9.6f} {100 * (cd - CD_REF) / CD_REF:+6.2f}% |"
        line += "".join(f"{100 * err[n] / CD_REF:+7.2f}" for n, _, _ in ZONES)
        if near:
            line += f" | {near['TE']:7d} {near['LE']:4d} {near['spig']:4d}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
