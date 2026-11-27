#!/usr/bin/env python3
"""Gmsh -> SU2 Euler steady -> shock detector -> refined Gmsh mesh.

Compatible with test-simone/Diagnostics/.../diamond_euler.geo (baseline geo)
and the original legacy diamondShockRefinement.geo (METHOD/SHOCK_FILE).
Original inputs are never modified.
"""
import argparse
import csv
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def checked_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label}: file non trovato: {path}')
    return path


def write_cfg(source, dst):
    data = source.read_text(encoding='utf-8')
    changes = {
        'MESH_FILENAME': 'mesh.su2',
        'CONV_FILENAME': 'Results/history',
        'RESTART_FILENAME': 'Results/restart_flow.dat',
        'SOLUTION_FILENAME': 'Results/restart_flow.dat',
        'VOLUME_FILENAME': 'Results/flow',
        'SURFACE_FILENAME': 'Results/surface_flow',
        'RESTART_SOL': 'NO',
        'TIME_DOMAIN': 'NO',
        'OUTPUT_FILES': '( RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV )',
    }
    for key, value in changes.items():
        pat = re.compile(r'(?im)^(\s*' + key + r'\s*=\s*)[^\r\n%]*')
        if pat.search(data):
            data = pat.sub(lambda m: m.group(1) + value, data)
        else:
            data += f'\n{key}= {value}\n'
    dst.write_text(data, encoding='utf-8')


def run(cmd, cwd, logfile, dry_run=False):
    command = shlex.join([str(arg) for arg in cmd])
    print(f'[{cwd.name}] $ {command}', flush=True)
    with logfile.open('a', encoding='utf-8') as out:
        out.write(f'\n$ {command}\n')
        if dry_run:
            out.write('DRY RUN\n')
            return
        try:
            with subprocess.Popen([str(arg) for arg in cmd], cwd=cwd,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as proc:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    out.write(line)
                result = proc.wait()
        except FileNotFoundError as exc:
            raise RuntimeError(f'Eseguibile non trovato: {cmd[0]}') from exc
        if result:
            raise RuntimeError(
                f'Comando fallito (exit {result}). Vedi {logfile}')


def arrays_from_shock(path):
    txt = path.read_text(encoding='utf-8')
    values = []
    for name in ('sx', 'sy'):
        match = re.search(
            r'\b' + name + r'\s*\[\s*\]\s*=\s*\{([^}]*)\}', txt, re.S)
        if match is None:
            raise ValueError(f'{path}: manca array {name}[]')
        values.append([float(s) for s in match.group(
            1).replace('\n', ' ').split(',') if s.strip()])
    xs, ys = values
    if not xs or len(xs) != len(ys):
        raise ValueError(f'{path}: sx[]/sy[] vuoti o lunghezze differenti')
    if not all(abs(x) < 1e30 and abs(y) < 1e30 for x, y in zip(xs, ys)):
        raise ValueError(f'{path}: punti non finiti')
    return list(zip(xs, ys))


def set_wall_size(text, hwall):
    if hwall is None:
        return text
    # Source GEO uses hWall = ...; legacy GEO may instead use h_wall.
    for name in ('hWall', 'h_wall'):
        pattern = re.compile(r'(?m)^(\s*' + name + r'\s*=\s*)[^;]+;')
        if pattern.search(text):
            return pattern.sub(lambda m: m.group(1) + str(hwall) + ';', text, count=1)
    print('AVVISO: --h-wall ignorato: nessuna variabile hWall/h_wall nel .geo')
    return text


def make_geometry(src_geo, dst_geo, cycle_number, prior_shock, args, legacy):
    geo = set_wall_size(src_geo.read_text(encoding='utf-8'), args.h_wall)
    if legacy:
        # Refinement is implemented by the existing GEO itself.
        dst_geo.write_text(geo, encoding='utf-8')
        return
    if cycle_number > 0:
        points = [(0.1, 0.1)] if args.dry_run else arrays_from_shock(
            prior_shock)
        # Append a Gmsh Distance + Threshold field. No modifications to source.
        lines = [
            '\n// === Automatic iterative refinement overlay ===',
            '// Points detected in previous cycle, not an anisotropic metric.',
            'shockTags[] = {};',
        ]
        for x, y in points:
            lines += [f'pt = newp; Point(pt) = {{{x:.16g}, {y:.16g}, 0, {args.h_min:.16g}}};',
                      'shockTags[] += {pt};']
        # High field IDs to avoid collision with fields in the Euler reference .geo.
        # Use point distance; Min preserves the baseline wall-size field [2].
        lines += [
            'Field[9001] = Distance;',
            'Field[9001].PointsList = {shockTags[]};',
            'Field[9002] = Threshold;',
            'Field[9002].InField = 9001;',
            f'Field[9002].SizeMin = {args.h_min:.16g};',
            f'Field[9002].SizeMax = {args.h_max:.16g};',
            f'Field[9002].DistMin = {args.shock_width:.16g};',
            f'Field[9002].DistMax = {args.shock_width + args.shock_ramp:.16g};',
            'Field[9003] = Min;',
            'Field[9003].FieldsList = {2, 9002};',
            'Background Field = 9003;',
            f'Mesh.MeshSizeMin = {args.h_min:.16g};',
            f'Mesh.MeshSizeMax = {args.h_max:.16g};',
        ]
        geo += '\n'.join(lines) + '\n'
    dst_geo.write_text(geo, encoding='utf-8')


def find_flow(cycle):
    d = cycle / 'Results'
    files = list(d.glob('flow*.vtu'))
    if not files:
        raise FileNotFoundError(
            f'Impossibile trovare il VTU della soluzione in {d}')
    steady = d / 'flow.vtu'
    if steady.exists():
        return steady
    return sorted(files, key=lambda p: p.stat().st_mtime_ns)[-1]


def main():
    parser = argparse.ArgumentParser(
        description='Raffinamento iterativo Euler 2D con Gmsh e SU2')
    parser.add_argument('--geo', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', '--cycles', type=int, default=3)
    parser.add_argument('--ncores', type=int, default=1)
    parser.add_argument('--h-wall', type=float, default=None)
    parser.add_argument('--h-min', type=float, default=0.01)
    parser.add_argument('--h-max', type=float, default=0.30)
    parser.add_argument('--shock-width', type=float, default=0.025,
                        help='Distanza dalla shock line a dimensione minima')
    parser.add_argument('--shock-ramp', type=float, default=0.10,
                        help='Distanza di transizione da h-min a h-max')
    parser.add_argument('--method', type=int, default=2,
                        help='Solo per .geo legacy: METHOD cicli successivi')
    parser.add_argument('--gmsh', default='gmsh')
    parser.add_argument('--su2', default='SU2_CFD')
    parser.add_argument('--mpi', default='mpirun')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.ncores < 1:
        parser.error('--iterations e --ncores devono essere >= 1')
    if min(args.h_min, args.h_max) <= 0 or args.h_min > args.h_max:
        parser.error('Richiesto 0 < --h-min <= --h-max')
    if args.shock_width < 0 or args.shock_ramp <= 0:
        parser.error('Richiesti --shock-width >= 0 e --shock-ramp > 0')

    geo = checked_file(args.geo, 'Geometria')
    cfg = checked_file(args.config, 'Configurazione SU2')
    detector = checked_file(args.detector, 'Rilevatore VTU')
    out_dir = Path(args.output).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(
            f'Cartella output non vuota: {out_dir}. Seleziona una cartella nuova.')
    out_dir.mkdir(parents=True, exist_ok=True)

    original_text = geo.read_text(encoding='utf-8')
    legacy = ('SHOCK_FILE' in original_text and 'METHOD' in original_text)
    print('Modalita geometria:', 'legacy con raffinamento interno' if legacy else
          'baseline con campo Distance + Threshold aggiunto ai cicli successivi')
    previous_shock = None
    csv_file = out_dir / 'cycles_log.csv'
    for i in range(args.iterations):
        cycle = out_dir / f'cycle_{i:02d}'
        cycle.mkdir()
        (cycle / 'Results').mkdir()
        log = cycle / 'run.log'
        for adjacent in geo.parent.iterdir():
            if adjacent.is_file() and adjacent.suffix.lower() in ('.geo', '.pos') and adjacent != geo:
                shutil.copy2(adjacent, cycle / adjacent.name)
        make_geometry(geo, cycle / 'geometry.geo', i,
                      previous_shock, args, legacy)
        write_cfg(cfg, cycle / 'run.cfg')
        shutil.copy2(detector, cycle / 'detector.py')
        if previous_shock is not None and not args.dry_run:
            shutil.copy2(previous_shock, cycle / 'shockPoints.geo')

        gmsh_cmd = [args.gmsh, 'geometry.geo', '-2', '-format', 'su2', '-o', 'mesh.su2',
                    '-v', '2']
        if legacy:
            gmsh_cmd += ['-setnumber', 'METHOD', str(0 if i == 0 else args.method),
                         '-setstring', 'OUTFILE', 'mesh.su2']
            if args.h_min:
                gmsh_cmd += ['-setnumber', 'h_min', str(args.h_min)]
            if args.h_wall is not None:
                gmsh_cmd += ['-setnumber', 'h_wall', str(args.h_wall)]
            if i > 0:
                gmsh_cmd += ['-setnumber', 'SHOCK_TYPE', '3',
                             '-setstring', 'SHOCK_FILE', 'shockPoints.geo']
        run(gmsh_cmd, cycle, log, args.dry_run)
        mesh = cycle / 'mesh.su2'
        if not args.dry_run and (not mesh.is_file() or mesh.stat().st_size == 0):
            raise RuntimeError(
                f'Gmsh non ha prodotto una mesh SU2 valida: {mesh}. Vedi {log}')

        su2_cmd = ([args.mpi, '-n', str(args.ncores), args.su2, 'run.cfg']
                   if args.ncores > 1 else [args.su2, 'run.cfg'])
        run(su2_cmd, cycle, log, args.dry_run)
        flow = cycle / 'Results/flow.vtu' if args.dry_run else find_flow(cycle)
        shock = cycle / 'shockPoints.geo'
        run([args.python, 'detector.py', '-VTUfile', str(flow), '-GEOfile', str(shock)],
            cycle, log, args.dry_run)
        if not args.dry_run:
            if not shock.exists():
                raise RuntimeError(
                    f'Il detector non ha creato {shock}. Controlla {log}')
            arrays_from_shock(shock)

        with csv_file.open('a', newline='', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            if i == 0:
                writer.writerow(['cycle', 'mesh_su2', 'flow_vtu', 'shock_geo'])
            writer.writerow([i, mesh, flow, shock])
        previous_shock = shock
        print(f'Fine ciclo {i:02d}: mesh={mesh}', flush=True)
    print(
        f'\nMesh finale simulata: {out_dir / f"cycle_{args.iterations-1:02d}" / "mesh.su2"}')
    print(f'Risultati: {out_dir}')


if __name__ == '__main__':
    main()
