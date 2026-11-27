"""
Convergenza del C_D al variare della complessita', per tutti i run di un caso.

Legge  runs/<caso>/<run>/cycles_log.csv  e  parametri.txt  di ogni run, e per ciascuno
calcola sulle ultime K iterazioni:
    - elementi medi
    - C_D medio, minimo e massimo (la dispersione del punto fisso)
    - C_D finale (ultima iterazione, quella convergita fino in fondo)
    - errore rispetto al riferimento (teoria urto-espansione: 0.011341)

Scrive:
    runs/<caso>/convergence_summary.csv     tabella riassuntiva
    runs/<caso>/convergence.png             |errore| vs elementi (log-log) e C_D vs elementi

Uso (dalla cartella test-simone):
    python3 plot_convergence.py                          tutti i run di diamond_euler
    python3 plot_convergence.py --last 4                 media sulle ultime 4 iterazioni
    python3 plot_convergence.py --runs hess_C2500 hess_C5000 goal_C10k
    python3 plot_convergence.py --help
"""
import argparse
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent          # cartella test-simone

# =============================================================================
# Parametri
# =============================================================================
parser = argparse.ArgumentParser(
    description="Convergenza del C_D al variare della complessita'")
parser.add_argument("--case", default="diamond_euler",
                    help="cartella in runs/ (default: %(default)s)")
parser.add_argument("--runs", nargs="+",
                    help="nomi dei run da usare (default: tutti)")
parser.add_argument("--last", type=int, default=3,
                    help="iterazioni finali su cui fare media e dispersione (default: %(default)s)")
parser.add_argument("--ref", type=float, default=0.011341,
                    help="C_D di riferimento (default: %(default)s, teoria urto-espansione)")
args = parser.parse_args()

case_dir = HERE / "runs" / args.case
if not case_dir.is_dir():
    raise SystemExit(f"Cartella non trovata: {case_dir}")


# =============================================================================
# Lettura di un run
# =============================================================================
def read_parameters(run_dir):
    """parametri.txt -> dizionario {nome: testo del valore}."""
    params = {}
    path = run_dir / "parametri.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if " = " in line:
                name, value = line.split(" = ", 1)
                params[name.strip()] = value.strip()
    return params


def read_log(run_dir):
    """cycles_log.csv -> lista di righe (dizionari con numeri dove possibile)."""
    rows = []
    with open(run_dir / "cycles_log.csv", newline="") as f:
        for row in csv.DictReader(f):
            clean = {}
            for key, value in row.items():
                try:
                    clean[key] = float(value)
                except (TypeError, ValueError):
                    clean[key] = None
            rows.append(clean)
    return rows


def summarize(run_dir, last):
    """Riassunto di un run: complessita', elementi e C_D sulle ultime `last` iterazioni."""
    params = read_parameters(run_dir)
    rows = [r for r in read_log(run_dir) if r.get("CD") is not None]
    if len(rows) < 2:
        return None                                    # run interrotto o solo mesh iniziale

    # la mesh iniziale (iterazione 0) non e' adattata: la escludiamo dalla media
    adapted = rows[1:]
    tail = adapted[-last:]

    # complessita' usata per costruire le mesh della coda (dalla riga precedente)
    targets = [r["metric_target"] for r in rows[:-1] if r.get("metric_target")]
    complexity = targets[-1] if targets else float("nan")

    cds = [r["CD"] for r in tail]
    elements = [r["elements"] for r in tail]
    return {
        "run": run_dir.name,
        "metric": params.get("metric", "hessian"),
        "complexity": complexity,
        "n_iter": len(rows),
        "elements": sum(elements) / len(elements),
        "CD_mean": sum(cds) / len(cds),
        "CD_min": min(cds),
        "CD_max": max(cds),
        "CD_final": rows[-1]["CD"],
        "used": len(tail),
    }


# =============================================================================
# Raccolta dei run
# =============================================================================
run_dirs = sorted(d for d in case_dir.iterdir()
                  if (d / "cycles_log.csv").is_file())
if args.runs:
    run_dirs = [d for d in run_dirs if d.name in args.runs]
if not run_dirs:
    raise SystemExit(f"Nessun run con cycles_log.csv in {case_dir}")

summary = []
for d in run_dirs:
    s = summarize(d, args.last)
    if s is None:
        print(f"  (salto {d.name}: meno di 2 iterazioni)")
        continue
    s["err_mean_pct"] = 100 * (s["CD_mean"] - args.ref) / args.ref
    s["err_final_pct"] = 100 * (s["CD_final"] - args.ref) / args.ref
    s["spread_pct"] = 100 * (s["CD_max"] - s["CD_min"]) / 2 / args.ref
    summary.append(s)

summary.sort(key=lambda s: (s["metric"], s["elements"]))

# --- tabella a schermo -------------------------------------------------------
print(
    f"\nC_D di riferimento: {args.ref}   (media sulle ultime {args.last} iterazioni adattate)\n")
header = f"{'run':<22}{'metrica':<9}{'complessita':>12}{'elementi':>10}{'C_D medio':>12}{'errore %':>10}{'+/- %':>8}{'C_D finale':>12}{'errore %':>10}"
print(header)
print("-" * len(header))
for s in summary:
    print(f"{s['run']:<22}{s['metric']:<9}{s['complexity']:>12.0f}{s['elements']:>10.0f}"
          f"{s['CD_mean']:>12.6f}{s['err_mean_pct']:>10.2f}{s['spread_pct']:>8.2f}"
          f"{s['CD_final']:>12.6f}{s['err_final_pct']:>10.2f}")

# --- tabella CSV -------------------------------------------------------------
summary_file = case_dir / "convergence_summary.csv"
columns = ["run", "metric", "complexity", "n_iter", "used", "elements", "CD_mean", "CD_min",
           "CD_max", "CD_final", "err_mean_pct", "spread_pct", "err_final_pct"]
with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary)
print(f"\nTabella: {summary_file}")


# =============================================================================
# Ordine di convergenza osservato
# =============================================================================
# In 2D h ~ N^(-1/2): se |errore| ~ N^(-q/2), in log-log la pendenza e' -q/2.
# Si stima con una retta ai minimi quadrati su log(elementi), log(|errore|).
def observed_order(points):
    pts = [(math.log(n), math.log(abs(e))) for n, e in points if e != 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    return -2 * slope                                   # ordine q rispetto a h


metrics = sorted(set(s["metric"] for s in summary))
print("\nOrdine osservato (rispetto a h, con h ~ N^-1/2):")
for m in metrics:
    pts = [(s["elements"], s["err_mean_pct"])
           for s in summary if s["metric"] == m]
    q = observed_order(pts)
    print(
        f"  {m:<9} {'(servono almeno 2 complessita)' if q is None else f'q = {q:.2f}'}")


# =============================================================================
# Grafico
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "\nmatplotlib non installato: grafico non creato (la tabella CSV c'e').")

COLORS = {"hessian": "#2a78d6", "goal": "#eb6834", "mixed": "#1baa8c"}
LABELS = {"hessian": "Hessiana (pressione)",
          "goal": "goal-oriented (C_D)", "mixed": "mista"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

for m in metrics:
    runs = [s for s in summary if s["metric"] == m]
    x = [s["elements"] for s in runs]
    color = COLORS.get(m, INK2)
    label = LABELS.get(m, m)

    # sinistra: |errore| percentuale, log-log, con la dispersione delle ultime iterazioni
    y = [abs(s["err_mean_pct"]) for s in runs]
    # barra = dispersione; in basso non deve scendere sotto zero (scala logaritmica)
    lo = [min(s["spread_pct"], 0.9 * abs(s["err_mean_pct"])) for s in runs]
    hi = [s["spread_pct"] for s in runs]
    ax1.errorbar(x, y, yerr=[lo, hi], fmt="-o", color=color, ms=6, mec="white", mew=1.5,
                 lw=2, capsize=3, elinewidth=1, label=label)

    # destra: C_D medio con min-max
    cd = [s["CD_mean"] for s in runs]
    err = [[s["CD_mean"] - s["CD_min"]
            for s in runs], [s["CD_max"] - s["CD_mean"] for s in runs]]
    ax2.errorbar(x, cd, yerr=err, fmt="-o", color=color, ms=6, mec="white", mew=1.5,
                 lw=2, capsize=3, elinewidth=1, label=label)

# rette di riferimento: ordine 1 e ordine 2 in h
all_x = [s["elements"] for s in summary]
all_y = [abs(s["err_mean_pct"]) for s in summary if s["err_mean_pct"] != 0]
if all_x and all_y:
    x0, x1 = min(all_x), max(all_x)
    y0 = max(all_y)
    for q, style in [(1, ":"), (2, "--")]:
        ax1.plot([x0, x1], [y0, y0 * (x1 / x0) ** (-q / 2)],
                 style, color=INK2, lw=1)
        ax1.text(x1, y0 * (x1 / x0) ** (-q / 2),
                 f"  ordine {q}", va="center", fontsize=8, color=INK2)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("numero di elementi")
ax1.set_ylabel("|errore sul C_D|  [%]")
ax1.set_title("Convergenza dell'errore", fontsize=10)
ax1.legend(loc="lower left")

ax2.axhline(args.ref, color=INK2, lw=1, ls=":")
ax2.text(max(all_x), args.ref, f"riferimento {args.ref}  ",
         ha="right", va="bottom", fontsize=8, color=INK2)
ax2.set_xscale("log")
ax2.set_xlabel("numero di elementi")
ax2.set_ylabel("C_D")
ax2.set_title(
    f"C_D (media e min-max sulle ultime {args.last} iterazioni)", fontsize=10)
ax2.legend(loc="lower right")

figure_file = case_dir / "convergence.png"
fig.savefig(figure_file, dpi=150)
print(f"Grafico: {figure_file}")
