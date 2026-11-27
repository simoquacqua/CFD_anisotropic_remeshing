"""
Ciclo di adattamento SU2 + Gmsh.

Organizzazione delle cartelle (tutte dentro test-simone):
    cases/<caso>/          i file di partenza del caso:
                             - un file .cfg  (SU2)
                             - un file .geo  (Gmsh, SENZA metrica)
                             - un file .su2  (mesh iniziale; facoltativo:
                               se manca, Gmsh la genera dal .geo)
    runs/<caso>/<prova>/   i risultati, creati dal programma

Si lancia nel Docker (servono SU2_CFD e gmsh), dalla cartella test-simone.

    python3 run_adapt.py                       usa i valori di DEFAULT qui sotto
    python3 run_adapt.py --run prova2 --n-iter 3 --complexity 10000
    python3 run_adapt.py --run studio --n-iter 4 --complexity 2000 4000 8000
    python3 run_adapt.py --run goal1 --metric goal --n-iter 3 --complexity 10000
    python3 run_adapt.py --help                elenco di tutti i parametri
"""
from adaptation.loop import adapt_loop
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # = cartella test-simone
# per poter fare "import adaptation"
sys.path.insert(0, str(HERE))


# =============================================================================
# VALORI DI DEFAULT (usati se non li cambi da terminale)
# =============================================================================
DEFAULT = {
    "case": "diamond_euler",    # cartella dentro cases/
    # nome della prova (cartella dentro runs/<caso>/)
    "run": "prova",
    # soluzioni SU2 (le mesh adattate sono n_iter - 1)
    "n_iter": 2,
    "complexity": [5000],       # un valore, oppure n_iter - 1 valori
    # "hessian" oppure "goal" (goal-oriented sul C_D, con l'aggiunto)
    "metric": "hessian",
    # campo per l'Hessiano: "Pressure", "Mach", "Density", ...
    "field": "Pressure",
    "p": 2,                     # norma Lp
    "h_min": 1e-3,              # taglia minima degli elementi
    "h_max": 0.3,               # taglia massima degli elementi
    "max_ar": 10,               # allungamento massimo degli elementi
    "su2": "SU2_CFD",           # in parallelo: "mpirun -n 4 SU2_CFD"
    # aggiunto discreto (solo per metric = "goal")
    "su2_adjoint": "SU2_CFD_AD",
    # residuo delle iterazioni intermedie (None = come nel cfg)
    "intermediate_residual": -6,
}


# =============================================================================
# Lettura dei parametri da terminale
# =============================================================================
parser = argparse.ArgumentParser(description="Ciclo di adattamento SU2 + Gmsh")
parser.add_argument("--case", default=DEFAULT["case"],
                    help="cartella del caso dentro cases/ (default: %(default)s)")
parser.add_argument("--run", default=DEFAULT["run"],
                    help="nome della prova, cioe' la cartella dei risultati (default: %(default)s)")
parser.add_argument("--n-iter", type=int, default=DEFAULT["n_iter"],
                    help="numero di soluzioni SU2 (default: %(default)s)")
parser.add_argument("--complexity", type=float, nargs="+", default=DEFAULT["complexity"],
                    help="complessita': un valore o n_iter-1 valori (default: %(default)s)")
parser.add_argument("--metric", default=DEFAULT["metric"], choices=["hessian", "goal"],
                    help="tipo di metrica (default: %(default)s)")
parser.add_argument("--field", default=DEFAULT["field"],
                    help="campo per l'Hessiano (default: %(default)s)")
parser.add_argument("--p", type=float, default=DEFAULT["p"],
                    help="norma Lp (default: %(default)s)")
parser.add_argument("--h-min", type=float, default=DEFAULT["h_min"],
                    help="taglia minima (default: %(default)s)")
parser.add_argument("--h-max", type=float, default=DEFAULT["h_max"],
                    help="taglia massima (default: %(default)s)")
parser.add_argument("--max-ar", type=float, default=DEFAULT["max_ar"],
                    help="aspect ratio massimo (default: %(default)s)")
parser.add_argument("--su2", default=DEFAULT["su2"],
                    help='comando SU2, es. "mpirun -n 4 SU2_CFD" (default: %(default)s)')
parser.add_argument("--su2-adjoint", default=DEFAULT["su2_adjoint"],
                    help="comando dell'aggiunto, es. \"mpirun -n 4 SU2_CFD_AD\" (default: %(default)s)")
parser.add_argument("--intermediate-residual", type=float, default=DEFAULT["intermediate_residual"],
                    help="CONV_RESIDUAL_MINVAL delle iterazioni intermedie (default: %(default)s)")
parser.add_argument("--keep-metric", action="store_true",
                    help="conserva i file metric.pos (pesanti)")
parser.add_argument("--overwrite", action="store_true",
                    help="permette di sovrascrivere una prova con lo stesso nome")
args = parser.parse_args()


# =============================================================================
# Trova i file del caso
# =============================================================================
def find_file(folder, extension, required=True):
    """Cerca nella cartella l'unico file con quell'estensione."""
    found = sorted(folder.glob("*" + extension))
    if len(found) == 1:
        return found[0]
    if len(found) == 0 and not required:
        return None
    if len(found) == 0:
        raise FileNotFoundError(f"Nessun file {extension} in {folder}")
    raise RuntimeError(f"Troppi file {extension} in {folder}: {[f.name for f in found]}. "
                       "Lasciane uno solo.")


case_dir = HERE / "cases" / args.case
if not case_dir.is_dir():
    raise FileNotFoundError(f"Cartella del caso non trovata: {case_dir}")

cfg_file = find_file(case_dir, ".cfg")
geo_file = find_file(case_dir, ".geo")
mesh_file = find_file(case_dir, ".su2", required=False)

# un solo valore di complessita' -> uguale per tutte le mesh
complexity = args.complexity[0] if len(
    args.complexity) == 1 else args.complexity

intermediate = {}
if args.intermediate_residual is not None:
    intermediate["CONV_RESIDUAL_MINVAL"] = args.intermediate_residual

print(f"Caso:  {args.case}   ->  risultati in runs/{args.case}/{args.run}")
print(f"  cfg:  {cfg_file.name}")
print(f"  geo:  {geo_file.name}")
print(f"  mesh: {mesh_file.name if mesh_file else '(generata da Gmsh)'}")
if args.metric == "goal":
    print(
        f"  metric=goal (aggiunto del C_D con '{args.su2_adjoint}'; --field e --p ignorati, p=1)")
else:
    print(f"  metric=hessian  field={args.field}  p={args.p}")
print(f"  n_iter={args.n_iter}  complexity={complexity}")
print(f"  h_min={args.h_min}  h_max={args.h_max}  max_ar={args.max_ar}")


# =============================================================================
# Ciclo
# =============================================================================
adapt_loop(
    cfg_file=cfg_file,
    geo_file=geo_file,
    initial_mesh=mesh_file,
    work_dir=HERE / "runs" / args.case / args.run,
    n_iter=args.n_iter,
    complexity=complexity,
    metric=args.metric,
    field=args.field,
    p=args.p,
    h_min=args.h_min,
    h_max=args.h_max,
    max_aspect_ratio=args.max_ar,
    su2_command=args.su2,
    su2_adjoint_command=args.su2_adjoint,
    intermediate_su2_options=intermediate,
    keep_metric=args.keep_metric,
    overwrite=args.overwrite,
)
