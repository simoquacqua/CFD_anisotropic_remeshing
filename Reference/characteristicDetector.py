#!/usr/bin/env python3
"""Indicatore di compressione caratteristica (Feng et al., 2024).

Implementa le eq. (4.2), (4.4), (4.6), (4.8)-(4.10) per celle
triangolari conformi SU2/VTU. L'estensione della scelta dei vicini
ai triangoli e la media aritmetica nodale->cella sono adattamenti,
non una replica della discretizzazione DG quadrilatera del paper.

Note sull'implementazione:
  - pesi (4.6) e criterio sinistra/destra come nel paper (default tol=0);
  - default W, M1, M2 = valori del paper (10, 1.2, 0.1);
  - --normalize divide il salto per a locale (NON e' nel paper);
  - --top-fraction marca la frazione theta di celle con indicatore piu' alto
    (strategia B dei casi stazionari);
  - campo FengRatio = salto / (M1*h + M2): >= 1 equivale a margine >= 0 ed e'
    confrontabile tra celle di dimensione diversa (utile su mesh grossolane);
  - campi diagnostici: FengMargin, FengWinningJump, FengMaxAbsJump, MachCell...

Esempio (dalla cartella Reference):
    python3 characteristicDetector.py --M1 0.1 --M2 0.005 --top-fraction 0.05

Dipendenze: numpy, vtk (vtk viene installato anche con pyvista).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def logistic(z):
    z = np.clip(z, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-z))


def load_triangular_grid(path):
    reader = vtk.vtkXMLUnstructuredGridReader()  # creates a reader for .vtu files
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()  #it returns a vtkUnstructuredGrid object which is made of 
    # connectivity matrix, variables, cells and points 
    if grid is None or grid.GetNumberOfCells() == 0:  # control on the number of cells
        raise ValueError(f"Mesh VTU vuota o non leggibile: {path}")
    n = grid.GetNumberOfCells()  # it gives the number of cells 
    types = np.array([grid.GetCellType(i) for i in range(n)], dtype=int)
    if not np.all(types == vtk.VTK_TRIANGLE):
        raise ValueError("Questo prototipo gestisce solo triangoli VTK (tipo 5).")
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(float)
    connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    tri = connectivity.reshape(n, 3).astype(np.int64)
    return grid, points, tri


def field(grid, key):
    arr = grid.GetPointData().GetArray(key)
    if arr is None:
        raise ValueError(f"Campo point-data non trovato: '{key}'")
    return vtk_to_numpy(arr).astype(float)


def compute_indicator(grid, points, tri, gamma=1.4, W=10.0, M1=1.2, M2=0.1,
                      Lref=1.0, tol=0.0, sign=1.0, normalize=False):
    """Restituisce valori per cella e diagnostica della mesh."""
    # SU2 esporta dati nodali: per ciascun triangolo utilizziamo
    # la media aritmetica dei tre valori nodali (approssimazione).
    rho = field(grid, "Density")[tri].mean(axis=1)
    pres = field(grid, "Pressure")[tri].mean(axis=1)
    vel = field(grid, "Velocity")[tri].mean(axis=1)
    if np.any(rho <= 0) or np.any(pres <= 0):
        raise ValueError("Densita' o pressione non positive: impossibile calcolare il suono.")
    a = np.sqrt(gamma * pres / rho)
    mach = np.linalg.norm(vel[:, :2], axis=1) / a

    # Controllo unita': se a~1 e |u|~2 il VTU e' adimensionale, se a~340 e' dimensionale.
    print(f"[unita'] rho medio={rho.mean():.5g}  p medio={pres.mean():.5g}  "
          f"a medio={a.mean():.5g}  Mach medio={mach.mean():.4f}  "
          f"Mach max={mach.max():.4f}")
    print(f"[unita'] velocita' media (u,v,w)={vel.mean(axis=0)}")

    # Le 4 famiglie genuine non lineari secondo (4.4),(4.10).
    lam = np.column_stack((vel[:, 0] - a, vel[:, 0] + a,
                           vel[:, 1] - a, vel[:, 1] + a))
    cent = points[tri, :2].mean(axis=1)
    p0, p1, p2 = (points[tri[:, j], :2] for j in range(3))
    e1, e2 = p1 - p0, p2 - p0
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if np.any(area <= 0):
        raise ValueError("Trovati triangoli degeneri con area nulla.")

    # Una coppia di triangoli e' vicina se condivide un intero lato.
    edge_to_cells = defaultdict(list)
    for e, (i, j, k) in enumerate(tri):
        for v0, v1 in ((i, j), (j, k), (k, i)):
            edge_to_cells[(min(v0, v1), max(v0, v1))].append(e)
    neighbors = [[] for _ in range(len(tri))]
    for face, ids in edge_to_cells.items():
        if len(ids) == 2:
            e, k = ids
            neighbors[e].append(k)
            neighbors[k].append(e)
        elif len(ids) > 2:
            raise ValueError(f"Faccia non-manifold: {face}, celle={ids}")

    values = np.zeros(len(tri), dtype=float)
    jump = np.zeros(len(tri), dtype=float)
    max_abs_jump = np.zeros(len(tri), dtype=float)
    h_values = np.zeros((len(tri), 2), dtype=float)
    # Diagnostica: quale direzione/famiglia controlla il massimo.
    best_margin = np.full(len(tri), -np.inf)
    best_h = np.zeros(len(tri))
    best_jump = np.zeros(len(tri))
    best_dir = np.full(len(tri), -1, dtype=int)
    best_family = np.full(len(tri), -1, dtype=int)
    best_ratio = np.full(len(tri), -1.0)
    missing_side_count = np.zeros(len(tri), dtype=int)

    for e, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        diffs = cent[nbrs] - cent[e]
        distances = np.linalg.norm(diffs, axis=1)
        if np.any(distances < 1e-14):
            raise ValueError(f"Centri di celle coincidenti vicino alla cella {e}")
        directions = diffs / distances[:, None]  # eq. (4.5), same-level
        ratios = area[nbrs] / (area[nbrs] + area[e])
        # eq. (4.9), reso adimensionale con la lunghezza di riferimento
        h_values[e] = np.max(np.abs(diffs), axis=0) / Lref
        candidates = []
        candidates_jump = []
        candidate_meta = []
        ratio_max = -np.inf
        for d in range(2):
            # Dimension splitting: famiglia x o y, eq. (4.10).
            families = (0, 1) if d == 0 else (2, 3)
            fam = list(families)
            # Vicini quasi perpendicolari alla direzione d vengono scartati.
            # Paper (sez. 4.1.2): sinistra se n_kd <= 0, destra se n_kd > 0 (tol=0).
            L = [j for j in range(len(nbrs)) if directions[j, d] <= -tol]
            R = [j for j in range(len(nbrs)) if directions[j, d] > tol]
            missing_side_count[e] += int(not L) + int(not R)
            # Nessun vicino sul lato: usa valore della cella corrente.
            # E' una scelta di chiusura per celle vicine al bordo.
            side_avgs = []
            for side in (L, R):
                if not side:
                    side_avgs.append(lam[e, fam].copy())
                    continue
                weighted_neighbors = []
                for j in side:
                    k = nbrs[j]
                    nd2 = directions[j, d] ** 2
                    wk = nd2 * ratios[j]
                    w0 = (1.0 - nd2) * (1.0 - ratios[j])  # paper eq. (4.6)
                    # eq. (4.6): media pesata tra E0 ed Ek.
                    weighted_neighbors.append(
                        (wk * lam[k, fam] + w0 * lam[e, fam]) / (wk + w0)
                    )
                # eq. (4.8): media sui vicini del medesimo lato.
                side_avgs.append(np.mean(weighted_neighbors, axis=0))
            # Salto (normalizzato con a solo se --normalize).
            scale = a[e] if normalize else 1.0
            s = sign * (side_avgs[0] - side_avgs[1]) / scale
            z = W * (s - M1 * h_values[e, d] - M2)
            candidates.extend(logistic(z).tolist())
            candidates_jump.extend(s.tolist())
            for f, sval in enumerate(s):
                ratio_max = max(ratio_max, float(sval) / (M1 * h_values[e, d] + M2))
                candidate_meta.append((float(sval - M1 * h_values[e, d] - M2),
                                       float(sval), float(h_values[e, d]), d, families[f]))
        values[e] = max(candidates)  # eq. (4.10)
        jump[e] = max(candidates_jump)
        max_abs_jump[e] = max(abs(x) for x in candidates_jump)
        margin, jump_winner, h_winner, d_winner, f_winner = max(candidate_meta, key=lambda t: t[0])
        best_margin[e] = margin
        best_jump[e] = jump_winner
        best_h[e] = h_winner
        best_dir[e] = d_winner
        best_family[e] = f_winner
        best_ratio[e] = max(ratio_max, -1.0)

    # Le celle senza vicini hanno margine -inf: lo sostituiamo per la scrittura VTU.
    best_margin = np.where(np.isfinite(best_margin), best_margin, -1.0)

    diagnostics = {
        "FengMargin": best_margin,
        "FengRatio": best_ratio,
        "FengWinningJump": best_jump,
        "FengWinningH": best_h,
        "FengDirection": best_dir,
        "FengFamily": best_family,
        "FengMaxAbsJump": max_abs_jump,
        "MissingSideCount": missing_side_count,
        "MachCell": mach,
    }
    return values, jump, h_values, np.array([len(n) for n in neighbors]), diagnostics


def add_cell_array(grid, name, values):
    array = numpy_to_vtk(np.asarray(values, dtype=np.float64), deep=True)
    array.SetName(name)
    grid.GetCellData().AddArray(array)


def main():
    default_input = Path(__file__).resolve().parents[1] / "TEST002" / "Results" / "flow_00000.vtu"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--W", type=float, default=10.0)
    parser.add_argument("--M1", type=float, default=1.2)
    parser.add_argument("--M2", type=float, default=0.1)
    parser.add_argument("--Lref", type=float, default=1.0,
                        help="lunghezza di riferimento (es. corda del profilo) per rendere h adimensionale")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="soglia su |n_d| sotto la quale un vicino e' escluso dallo stencil L/R")
    parser.add_argument("--flip-sign", action="store_true",
                        help="inverte il segno del salto (s = R - L) per verificare la convenzione")
    parser.add_argument("--normalize", action="store_true",
                        help="divide il salto per la velocita' del suono locale (NON e' nel paper)")
    parser.add_argument("--top-fraction", type=float, default=None,
                        help="come nei casi stazionari del paper (strategia B): marca la frazione "
                             "theta di celle con indicatore piu' alto, invece della soglia fissa")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold deve essere compreso tra 0 e 1.")
    if args.Lref <= 0:
        parser.error("--Lref deve essere positivo.")
    inp = args.input.resolve()
    if not inp.exists():
        parser.error(f"File VTU non trovato: {inp}")
    out = args.output or inp.with_name("shock_indicator_feng.vtu")
    out = out.resolve()
    if out == inp:
        parser.error("Il file di output non puo' sovrascrivere l'input.")

    grid, coords, triangles = load_triangular_grid(inp)
    indicator, jumps, h, n_neighbors, diagnostics = compute_indicator(
        grid, coords, triangles, args.gamma, args.W, args.M1, args.M2,
        Lref=args.Lref, tol=args.tol, sign=-1.0 if args.flip_sign else 1.0,
        normalize=args.normalize
    )
    add_cell_array(grid, "FengIndicator", indicator)
    if args.top_fraction is not None:
        cut = np.quantile(indicator, 1.0 - args.top_fraction)
        detected = (indicator >= cut) & (indicator > 0)
        print(f"Strategia B: frazione {args.top_fraction:g} -> soglia indicatore {cut:.5f}")
    else:
        detected = indicator >= args.threshold
    add_cell_array(grid, "FengDetected", detected.astype(float))
    add_cell_array(grid, "CharacteristicJump", jumps)
    add_cell_array(grid, "h_x", h[:, 0])
    add_cell_array(grid, "h_y", h[:, 1])
    add_cell_array(grid, "NeighborCount", n_neighbors)
    for name, vals in diagnostics.items():
        add_cell_array(grid, name, vals)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(out))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise IOError(f"Errore nella scrittura di {out}")

    winning_jump = diagnostics["FengWinningJump"]
    winning_h = diagnostics["FengWinningH"]
    margin = diagnostics["FengMargin"]
    print(f"Triangoli: {grid.GetNumberOfCells()}")
    print(f"Parametri: W={args.W:g} M1={args.M1:g} M2={args.M2:g} "
          f"Lref={args.Lref:g} tol={args.tol:g} segno={'R-L' if args.flip_sign else 'L-R'}")
    print(f"Celle con indicatore >= {args.threshold:g}: {np.count_nonzero(indicator >= args.threshold)}")
    print(f"Indicatore: min={indicator.min():.5f}, max={indicator.max():.5f}")
    jlabel = "normalizzato su a" if args.normalize else "grezzo"
    print(f"Salto vincente ({jlabel}): max={winning_jump.max():.6f}; "
          f"mediana={np.median(winning_jump):.6f}")
    print(f"max|salto| su tutte le famiglie: {diagnostics['FengMaxAbsJump'].max():.6f}")
    print(f"h vincente (adim.): mediana={np.median(winning_h):.6f}; max={winning_h.max():.6f}")
    print(f"Margine = salto - M1*h - M2: max={np.max(margin):.6f}")
    ratio = diagnostics["FengRatio"]
    print(f"FengRatio = salto/(M1*h+M2): max={ratio.max():.4f}; celle con ratio >= 1: "
          f"{np.count_nonzero(ratio >= 1.0)}")
    print(f"Celle con almeno un lato direzionale senza vicino: "
          f"{np.count_nonzero(diagnostics['MissingSideCount'])}")
    print(f"File generato: {out}")
    print("In ParaView: colora per Cell Data > FengMargin (scala centrata su 0)")
    print("oppure FengRatio (>= 1 = sopra soglia) / FengIndicator / FengMaxAbsJump.")
    print("NOTA: test della formula di Feng con adattamento triangolare;")
    print("non e' una validazione del metodo DG originale.")


if __name__ == "__main__":
    main()