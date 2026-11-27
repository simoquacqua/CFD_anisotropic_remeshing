#!/usr/bin/env python3
"""Seleziona le celle con il segnale Feng piu' alto e allarga la regione.

Esempio (da test-giorgio/Reference):
    python select_feng_top_cells.py \\
        --input ../TEST005_BASELINE/Results/feng_diagnostic.vtu \\
        --field FengRatio --percent 5 --min-value 1.0 --rings 2 \\
        --exclude-boundary \\
        --output ../TEST005_BASELINE/Results/feng_region.vtu

Differenze rispetto alla versione precedente:
  - --field sceglie il campo di ranking. FengRatio (o FengMargin) e' meglio di
    FengIndicator, che con la logistica si satura a 1 e rende arbitrario il
    ranking (parita' risolte con l'indice di cella);
  - --min-value: seleziona solo celle sopra una soglia minima, cosi' se non ci
    sono urti non viene selezionato comunque il P% (FengRatio >= 1 equivale a
    FengMargin >= 0);
  - --exclude-boundary: ignora le celle con meno di 3 vicini (profilo/bordi),
    che altrimenti consumano il budget con le singolarita' geometriche;
  - --rings N: allarga la regione di N anelli di celle (condivisione di
    almeno un vertice). Il campo RingLevel vale 0 sul nucleo, 1..N sugli anelli.

L'output contiene solo i triangoli selezionati.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import (numpy_to_vtk, numpy_to_vtkIdTypeArray,
                                           vtk_to_numpy)


def dilate_levels(core, tri, n_cells, rings):
    """Livello di anello per ogni cella: 0 nucleo, 1..rings anelli, -1 fuori."""
    level = np.full(n_cells, -1, dtype=int)
    level[core] = 0
    vertex_to_cells = defaultdict(list)
    for c, verts in enumerate(tri):
        for v in verts:
            vertex_to_cells[int(v)].append(c)
    frontier = np.asarray(core, dtype=np.int64)
    for r in range(1, rings + 1):
        if frontier.size == 0:
            break
        verts = np.unique(tri[frontier].ravel())
        candidates = np.unique(np.concatenate(
            [np.asarray(vertex_to_cells[int(v)], dtype=np.int64) for v in verts]))
        new = candidates[level[candidates] < 0]
        level[new] = r
        frontier = new
    return level


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path,
                        help='VTU prodotto da characteristicDetector.py')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--percent', type=float, default=5.0,
                        help='Percentuale di celle del nucleo, in (0,100]')
    parser.add_argument('--field', default='FengIndicator',
                        help="campo di ranking in Cell Data (consigliato: FengRatio o FengMargin)")
    parser.add_argument('--min-value', type=float, default=None,
                        help='seleziona solo celle con campo >= valore (es. 1.0 per FengRatio)')
    parser.add_argument('--exclude-boundary', action='store_true',
                        help='ignora celle con NeighborCount < 3')
    parser.add_argument('--rings', type=int, default=0,
                        help='anelli di celle con cui allargare la regione')
    args = parser.parse_args()
    if not (0 < args.percent <= 100):
        parser.error('--percent deve essere maggiore di 0 e non superiore a 100')
    if args.rings < 0:
        parser.error('--rings deve essere >= 0')
    if not args.input.is_file():
        parser.error(f'Input inesistente: {args.input}')
    output = args.output or args.input.with_name('feng_top5_cells.vtu')
    if output.resolve() == args.input.resolve():
        parser.error('Il file di output deve essere diverso da quello di input')

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(args.input))
    reader.Update()
    grid = reader.GetOutput()
    n = grid.GetNumberOfCells()
    if n == 0:
        raise RuntimeError('La mesh non contiene celle')
    types = np.array([grid.GetCellType(i) for i in range(n)], dtype=int)
    if not np.all(types == vtk.VTK_TRIANGLE):
        raise RuntimeError('Sono gestiti solo triangoli VTK (tipo 5).')

    arr = grid.GetCellData().GetArray(args.field)
    if arr is None:
        raise RuntimeError(f"Campo '{args.field}' non trovato in Cell Data. "
                           "Rilancia characteristicDetector.py (la versione aggiornata scrive FengRatio).")
    score = np.asarray(vtk_to_numpy(arr), dtype=float)
    if not np.isfinite(score).all():
        raise RuntimeError(f"{args.field} contiene NaN o Inf")

    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(float)
    tri = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(n, 3).astype(np.int64)

    eligible = np.ones(n, dtype=bool)
    if args.exclude_boundary:
        nc = grid.GetCellData().GetArray('NeighborCount')
        if nc is None:
            raise RuntimeError('NeighborCount non trovato: serve per --exclude-boundary')
        eligible &= np.asarray(vtk_to_numpy(nc)) >= 3
    if args.min_value is not None:
        eligible &= score >= args.min_value
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise RuntimeError('Nessuna cella soddisfa i criteri (prova ad abbassare --min-value).')

    k = min(n_eligible, max(1, math.ceil(n * args.percent / 100)))
    ranked = np.where(eligible, score, -np.inf)
    # Ordine decrescente, tie-break per indice di cella: selezione deterministica.
    core = np.argsort(-ranked, kind='stable')[:k].astype(np.int64)

    level = dilate_levels(core, tri, n, args.rings)
    chosen = np.nonzero(level >= 0)[0].astype(np.int64)

    ring_array = numpy_to_vtk(level.astype(np.float64), deep=True)
    ring_array.SetName('RingLevel')
    grid.GetCellData().AddArray(ring_array)

    node = vtk.vtkSelectionNode()
    node.SetFieldType(vtk.vtkSelectionNode.CELL)
    node.SetContentType(vtk.vtkSelectionNode.INDICES)
    node.SetSelectionList(numpy_to_vtkIdTypeArray(chosen, deep=True))
    selection = vtk.vtkSelection()
    selection.AddNode(node)
    extract = vtk.vtkExtractSelection()
    extract.SetInputData(0, grid)
    extract.SetInputData(1, selection)
    extract.Update()
    result = vtk.vtkUnstructuredGrid()
    result.ShallowCopy(extract.GetOutput())

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(result)
    if writer.Write() != 1:
        raise OSError(f'Impossibile scrivere {output}')

    p0, p1, p2 = (points[tri[:, j], :2] for j in range(3))
    e1, e2 = p1 - p0, p2 - p0
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    print(f'Celle totali: {n}  (idonee al ranking: {n_eligible})')
    print(f'Campo di ranking: {args.field}: min={score.min():.6g}, max={score.max():.6g}')
    print(f'Nucleo: {k} celle ({100*k/n:.2f}%), campo minimo nel nucleo {score[core].min():.6g}')
    if args.rings:
        for r in range(args.rings + 1):
            print(f'  RingLevel {r}: {int(np.count_nonzero(level == r))} celle')
    print(f'Regione totale: {chosen.size} celle ({100*chosen.size/n:.2f}%), '
          f'area {100*area[chosen].sum()/area.sum():.2f}% del dominio')
    print(f'Celle effettivamente scritte: {result.GetNumberOfCells()}')
    print(f'Output: {output.resolve()}')
    print("Nota: in caso di parita' sul valore di soglia, il ranking usa l'indice cella.")


if __name__ == '__main__':
    main()