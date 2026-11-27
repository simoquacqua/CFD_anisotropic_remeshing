"""
loop.py - Il ciclo di adattamento:  SU2 -> metrica -> Gmsh -> SU2 -> ...

Si usa chiamando adapt_loop(...) (vedi l'esempio test-simone/run_adapt.py).

Cartelle create:
    work_dir/
        parametri.txt       i parametri usati
        cycles_log.csv      una riga per iterazione (elementi, CD, CL, tempi, ...)
        iter_00/            mesh iniziale + soluzione SU2
        iter_01/            mesh adattata + soluzione SU2
        ...
"""
import csv
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import core, files


# =============================================================================
# Funzione principale
# =============================================================================

def adapt_loop(cfg_file, geo_file, work_dir,
               initial_mesh=None,
               n_iter=3,
               complexity=10000,
               metric="hessian",
               field="Pressure",
               p=2,
               h_min=1e-3,
               h_max=0.3,
               max_aspect_ratio=10,
               su2_command="SU2_CFD",
               gmsh_command="gmsh",
               su2_adjoint_command="SU2_CFD_AD",
               intermediate_su2_options=None,
               keep_metric=False,
               overwrite=False):
    """Esegue il ciclo di adattamento.

    Parametri
    ---------
    cfg_file        file .cfg di SU2 di partenza (non viene modificato)
    geo_file        file .geo di partenza, senza metrica (non viene modificato)
    work_dir        cartella dove mettere tutti i risultati
    initial_mesh    mesh .su2 iniziale; se None la genera Gmsh dal .geo
    n_iter          quante volte risolvere con SU2 (le mesh adattate sono n_iter - 1)
    complexity      complessita' delle mesh adattate (~ meta' del numero di triangoli):
                    un numero (uguale per tutte) o una lista di n_iter - 1 numeri
    metric          "hessian" (errore di interpolazione del campo `field`) oppure
                    "goal" (errore sul C_D, con l'aggiunto; usa sempre p = 1)
    field           campo di cui calcolare l'Hessiano ("Pressure", "Mach", ...)
    p               norma Lp dell'errore di interpolazione
    h_min, h_max    taglia minima e massima degli elementi (None = nessun limite)
    max_aspect_ratio  allungamento massimo degli elementi (None = nessun limite)
    su2_command     comando di SU2, es. "SU2_CFD" o "mpirun -n 4 SU2_CFD"
    gmsh_command    comando di Gmsh
    su2_adjoint_command  comando dell'aggiunto discreto (solo per metric="goal")
    intermediate_su2_options  opzioni SU2 solo per le iterazioni non finali,
                    es. {"CONV_RESIDUAL_MINVAL": -6}
    keep_metric     True = conserva i file metric.pos (pesano ~10 MB l'uno)
    overwrite       True = permette di riusare una work_dir gia' esistente

    Restituisce la lista delle righe scritte in cycles_log.csv.
    """
    cfg_file = Path(cfg_file).resolve()
    geo_file = Path(geo_file).resolve()
    work_dir = Path(work_dir).resolve()

    # --- controlli iniziali --------------------------------------------------
    if not cfg_file.is_file():
        raise FileNotFoundError(cfg_file)
    if not geo_file.is_file():
        raise FileNotFoundError(geo_file)
    geo_text = geo_file.read_text()
    if "Background Mesh" in geo_text:
        raise ValueError(
            f"{geo_file} contiene gia' una metrica: usa il .geo originale")

    # complessita': una per ogni mesh adattata
    if isinstance(complexity, (int, float)):
        complexity = [complexity] * (n_iter - 1)
    if len(complexity) != n_iter - 1:
        raise ValueError(
            f"complexity deve avere {n_iter - 1} valori (uno per mesh adattata)")

    if metric not in ("hessian", "goal"):
        raise ValueError(
            f"metric deve essere 'hessian' o 'goal', non '{metric}'")
    programs = [su2_command.split()[0], gmsh_command.split()[0]]
    if metric == "goal":
        programs.append(su2_adjoint_command.split()[0])
    for program in programs:
        if shutil.which(program) is None:
            raise FileNotFoundError(f"Programma '{program}' non trovato: "
                                    "lancia lo script dentro il Docker con SU2 e Gmsh")
    if (work_dir / "iter_00").exists() and not overwrite:
        raise RuntimeError(f"{work_dir} contiene gia' dei risultati: "
                           "cambia work_dir oppure usa overwrite=True")
    work_dir.mkdir(parents=True, exist_ok=True)

    if intermediate_su2_options is None:
        intermediate_su2_options = {}

    # salva i parametri usati, per ricordarsi come e' stato fatto il run
    with open(work_dir / "parametri.txt", "w") as f:
        for name, value in [("cfg_file", cfg_file), ("geo_file", geo_file),
                            ("initial_mesh", initial_mesh), ("n_iter", n_iter),
                            ("complexity", complexity), ("metric", metric),
                            ("field", field), ("p", p), ("h_min", h_min),
                            ("h_max", h_max), ("max_aspect_ratio", max_aspect_ratio),
                            ("su2_command", su2_command),
                            ("su2_adjoint_command", su2_adjoint_command),
                            ("intermediate_su2_options", intermediate_su2_options)]:
            f.write(f"{name} = {value}\n")

    # --- mesh iniziale -------------------------------------------------------
    first_dir = work_dir / "iter_00"
    first_dir.mkdir(exist_ok=True)
    if initial_mesh is not None:
        shutil.copy(initial_mesh, first_dir / "mesh.su2")
    else:
        print("Genero la mesh iniziale con Gmsh...")
        run_gmsh(geo_text, first_dir, "base.geo", gmsh_command)

    # --- ciclo ---------------------------------------------------------------
    log_rows = []
    for k in range(n_iter):
        iter_dir = work_dir / f"iter_{k:02d}"
        is_last = (k == n_iter - 1)

        n_elements, n_nodes = files.su2_mesh_counts(iter_dir / "mesh.su2")
        print(
            f"\n=== Iterazione {k}: {n_elements} elementi, {n_nodes} nodi ===")

        # 1) soluzione SU2
        if is_last:
            extra_options = {}
        else:
            extra_options = intermediate_su2_options
        history, su2_time = run_su2(
            cfg_file, iter_dir, su2_command, extra_options)

        row = {"iteration": k,
               "nodes": n_nodes,
               "elements": n_elements,
               "CD": history["CD"],
               "CL": history["CL"],
               "inner_iter": int(history["Inner_Iter"]),
               "rms_rho": history["rms[Rho]"],
               "su2_s": round(su2_time, 1)}
        print(f"  CD = {row['CD']:.6f}   CL = {row['CL']:.2e}   "
              f"iterazioni = {row['inner_iter']}   ({su2_time:.0f} s)")

        # 2) metrica e 3) nuova mesh (non serve dopo l'ultima soluzione)
        if not is_last:
            next_dir = work_dir / f"iter_{k + 1:02d}"
            next_dir.mkdir(exist_ok=True)

            # 2a) aggiunto del C_D (solo per la metrica goal-oriented)
            if metric == "goal":
                print("  SU2 aggiunto...")
                adjoint_time, res_final, res_min = run_su2_adjoint(cfg_file, iter_dir,
                                                                   su2_adjoint_command)
                row["adjoint_s"] = round(adjoint_time, 1)
                row["adjoint_res"] = round(res_final, 2)
                print(f"  aggiunto calcolato ({adjoint_time:.0f} s), residuo finale {res_final:.2f} "
                      f"(minimo {res_min:.2f})")
                if res_final > res_min + 1:
                    print("  ATTENZIONE: il residuo dell'aggiunto e' risalito di oltre un ordine "
                          "di grandezza: l'aggiunto sta divergendo, la metrica goal non e' affidabile")

            target = complexity[k]
            print(f"  Metrica '{metric}' con complessita' {target}...")
            metric_info = compute_metric(iter_dir, next_dir / "metric.pos", target,
                                         metric, field, p, h_min, h_max, max_aspect_ratio)
            row.update(metric_info)

            print("  Gmsh...")
            adapted_geo = files.make_adapted_geo(geo_text, "metric.pos",
                                                 h_min, h_max, max_aspect_ratio)
            row["gmsh_s"] = round(
                run_gmsh(adapted_geo, next_dir, "adapted.geo", gmsh_command), 1)

            if not keep_metric:
                (next_dir / "metric.pos").unlink()

        log_rows.append(row)
        write_log(work_dir / "cycles_log.csv", log_rows)

    print(f"\nFinito. Riepilogo in {work_dir / 'cycles_log.csv'}")
    return log_rows


# =============================================================================
# I singoli passi
# =============================================================================

def run_command(command, folder, log_file):
    """Lancia un programma nella cartella `folder` e salva l'output in log_file.
    Se il programma fallisce, mostra le ultime righe del log. Restituisce il tempo (s)."""
    start = time.time()
    with open(log_file, "w") as log:
        result = subprocess.run(command, cwd=folder,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        last_lines = Path(log_file).read_text(
            errors="replace").splitlines()[-20:]
        raise RuntimeError(f"Errore in {' '.join(command)} (cartella {folder}).\n"
                           "Ultime righe del log:\n" + "\n".join(last_lines))
    return time.time() - start


def run_su2(cfg_file, iter_dir, su2_command, extra_options):
    """Scrive il .cfg nella cartella dell'iterazione e lancia SU2.
    Restituisce (ultima riga di history.csv, tempo)."""
    (iter_dir / "Results").mkdir(exist_ok=True)

    # opzioni che il ciclo deve controllare (nomi dei file, formato di output)
    options = {
        "MESH_FILENAME": "mesh.su2",
        "MESH_FORMAT": "SU2",
        "RESTART_SOL": "NO",
        "TABULAR_FORMAT": "CSV",
        "CONV_FILENAME": "Results/history",
        "VOLUME_FILENAME": "Results/flow",
        "SURFACE_FILENAME": "Results/surface_flow",
        # il restart serve all'aggiunto (metrica goal), che riparte da questa soluzione
        "OUTPUT_FILES": "( RESTART, PARAVIEW, SURFACE_CSV )",
        "RESTART_FILENAME": "Results/restart_flow.dat",
        "SOLUTION_FILENAME": "Results/restart_flow.dat",
        "VOLUME_OUTPUT": "( COORDINATES, SOLUTION, PRIMITIVE, RESIDUAL, LIMITER )",
    }
    options.update(extra_options)

    cfg_text = files.set_cfg_options(cfg_file.read_text(), options)
    (iter_dir / cfg_file.name).write_text(cfg_text)

    flow_file = iter_dir / "Results" / "flow.vtu"
    if flow_file.exists():
        flow_file.unlink()                    # per non rileggere un vecchio risultato

    elapsed = run_command(su2_command.split() +
                          [cfg_file.name], iter_dir, iter_dir / "su2.log")
    if not flow_file.is_file():
        raise FileNotFoundError(f"SU2 non ha scritto {flow_file}")

    history = files.read_history(iter_dir / "Results" / "history.csv")
    return history, elapsed


def run_su2_adjoint(cfg_file, iter_dir, su2_adjoint_command):
    """Lancia l'aggiunto discreto del C_D sulla soluzione diretta di iter_dir.
    Legge Results/restart_flow.dat e scrive Results/adjoint.vtu.
    Restituisce il tempo impiegato."""
    options = {
        # stesse impostazioni di mesh del diretto
        "MESH_FILENAME": "mesh.su2",
        "MESH_FORMAT": "SU2",
        "RESTART_SOL": "NO",
        "TABULAR_FORMAT": "CSV",
        # aggiunto del C_D
        "MATH_PROBLEM": "DISCRETE_ADJOINT",
        "OBJECTIVE_FUNCTION": "DRAG",
        "CONV_FIELD": "RMS_ADJ_DENSITY",
        # stabilita' dell'aggiunto: il limitatore di Venkatakrishnan non e' derivabile,
        # e la sua derivata puo' far divergere l'iterazione dell'aggiunto -> lo si congela
        "FROZEN_LIMITER_DISC": "YES",
        "CFL_ADAPT": "NO",
        # per la metrica basta un aggiunto a -7; ITER fa da tetto di sicurezza
        "CONV_RESIDUAL_MINVAL": "-7",
        "ITER": "3000",
        # da dove leggere la soluzione diretta
        "SOLUTION_FILENAME": "Results/restart_flow.dat",
        # dove scrivere: nomi DIVERSI dal diretto, cosi' non si sovrascrivono
        "VOLUME_ADJ_FILENAME": "Results/adjoint",
        "SURFACE_ADJ_FILENAME": "Results/surface_adjoint",
        "RESTART_ADJ_FILENAME": "Results/restart_adj.dat",
        "CONV_FILENAME": "Results/history_adjoint",
        "OUTPUT_FILES": "( PARAVIEW, SURFACE_CSV )",
                "VOLUME_OUTPUT": "( COORDINATES, SOLUTION, PRIMITIVE, RESIDUAL, LIMITER )",
        # grandezze stampate: quelle del diretto (DRAG, AERO_COEFF) l'aggiunto non le conosce
        "SCREEN_OUTPUT": "( INNER_ITER, RMS_ADJ_DENSITY )",
        "HISTORY_OUTPUT": "( ITER, RMS_RES )",
    }
    cfg_text = files.set_cfg_options(cfg_file.read_text(), options)
    (iter_dir / "adjoint.cfg").write_text(cfg_text)

    if not (iter_dir / "Results" / "restart_flow.dat").is_file():
        raise FileNotFoundError(f"Manca {iter_dir / 'Results' / 'restart_flow.dat'}: "
                                "il diretto non ha scritto il restart")
    adjoint_file = iter_dir / "Results" / "adjoint.vtu"
    if adjoint_file.exists():
        adjoint_file.unlink()

    elapsed = run_command(su2_adjoint_command.split() + ["adjoint.cfg"], iter_dir,
                          iter_dir / "su2_adjoint.log")
    if not adjoint_file.is_file():
        written = sorted(f.name for f in iter_dir.rglob("*.vtu"))
        raise FileNotFoundError(f"L'aggiunto non ha scritto {adjoint_file}. "
                                f"File .vtu presenti: {written}. Controlla su2_adjoint.log")
    final_res, min_res = adjoint_residual(
        iter_dir / "Results" / "history_adjoint.csv")
    return elapsed, final_res, min_res


def adjoint_residual(history_file):
    """Residuo finale e minimo dell'aggiunto (colonna rms[A_Rho] di history_adjoint.csv)."""
    if not history_file.is_file():
        return float("nan"), float("nan")
    with open(history_file, newline="") as f:
        rows = list(csv.reader(f))
    names = [n.strip().strip('"') for n in rows[0]]
    if "rms[A_Rho]" not in names:
        return float("nan"), float("nan")
    j = names.index("rms[A_Rho]")
    values = [float(r[j]) for r in rows[1:] if len(r) > j]
    return values[-1], min(values)


def compute_metric(iter_dir, pos_file, target, metric, field, p, h_min, h_max, max_aspect_ratio):
    """Legge la soluzione, calcola la metrica, la normalizza e la scrive in pos_file.
    Restituisce un dizionario con le informazioni per il log."""
    start = time.time()
    flow = files.read_vtu(iter_dir / "Results" / "flow.vtu")
    points = flow["points"]
    triangles = flow["triangles"]

    # --- metrica grezza M0 (qui si aggiungeranno altre metriche) -------------
    if metric == "hessian":
        values = files.get_field(flow, field)
        M0, n_invalid = core.hessian_metric(points, triangles, values, p)
        if n_invalid > 0:
            print(
                f"    attenzione: {n_invalid} nodi senza Hessiano (usato H = 0)")
    elif metric == "goal":
        # l'aggiunto: stessa mesh, stessi nodi nello stesso ordine
        adjoint = files.read_vtu(iter_dir / "Results" / "adjoint.vtu")
        if adjoint["points"].shape != points.shape or not np.allclose(adjoint["points"], points):
            raise ValueError(
                "adjoint.vtu e flow.vtu non hanno gli stessi nodi")

        # soluzione diretta (-> 8 flussi: "dove la mesh sbaglia")
        rho = files.get_field(flow, "Density")
        rho_u = files.get_field(flow, "Momentum_x")
        rho_v = files.get_field(flow, "Momentum_y")
        rho_E = files.get_field(flow, "Energy")
        # aggiunto, nell'ordine delle equazioni (-> gradienti: "quanto conta per il C_D")
        psi = [files.get_field(adjoint, "Adjoint_Density"),
               files.get_field(adjoint, "Adjoint_Momentum_x"),
               files.get_field(adjoint, "Adjoint_Momentum_y"),
               files.get_field(adjoint, "Adjoint_Energy")]

        M0 = core.goal_oriented_metric(
            points, triangles, rho, rho_u, rho_v, rho_E, psi)

    else:
        raise ValueError(f"Metrica '{metric}' non implementata")

    # --- normalizzazione e limiti (uguale per tutte le metriche) -------------
    M, D = core.normalize_metric(
        M0, points, triangles, target, h_min, h_max, max_aspect_ratio)
    files.write_metric_pos(pos_file, points, triangles, M)

    return {"metric_target": target,
            "metric_complexity": round(core.metric_complexity(M, points, triangles), 1),
            "metric_D": D,
            "metric_s": round(time.time() - start, 1)}


def run_gmsh(geo_text, folder, geo_name, gmsh_command):
    """Scrive il .geo nella cartella e genera mesh.su2 con Gmsh. Restituisce il tempo."""
    (folder / geo_name).write_text(geo_text)
    mesh_file = folder / "mesh.su2"
    if mesh_file.exists():
        mesh_file.unlink()

    command = gmsh_command.split(
    ) + [geo_name, "-2", "-format", "su2", "-o", "mesh.su2"]
    elapsed = run_command(command, folder, folder / "gmsh.log")

    if not mesh_file.is_file():
        raise FileNotFoundError(f"Gmsh non ha scritto {mesh_file}")
    n_elements, _ = files.su2_mesh_counts(mesh_file)
    if n_elements < 1000:
        raise RuntimeError(
            f"Mesh sospetta: solo {n_elements} elementi in {mesh_file}")
    return elapsed


def write_log(path, rows):
    """Scrive cycles_log.csv (riscritto a ogni iterazione, cosi' e' sempre aggiornato)."""
    columns = ["iteration", "nodes", "elements", "CD", "CL", "inner_iter", "rms_rho", "su2_s",
               "adjoint_s", "adjoint_res", "metric_target", "metric_complexity", "metric_D", "metric_s", "gmsh_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
