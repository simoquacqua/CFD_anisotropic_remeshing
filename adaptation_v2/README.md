# adaptation — adattamento anisotropo SU2 + Gmsh

Ciclo  **SU2 → metrica → Gmsh → SU2 → …**  in una sola funzione: `adapt_loop`.

## Come si usa
Metti i file di partenza in `test-simone/cases/<nome_caso>/` (un .cfg, un .geo, facoltativa una mesh .su2), scegli caso e parametri in `test-simone/run_adapt.py` e nel Docker (dalla cartella `test-simone`):
```
python3 run_adapt.py
```
I risultati vanno in `runs/<nome_caso>/<nome_prova>/`. Il file da guardare per primo è
`cycles_log.csv`: una riga per iterazione con elementi, CD, CL, iterazioni SU2,
complessità richiesta e ottenuta, tempi.

## I file (da leggere in quest'ordine)
| File | Cosa fa |
|---|---|
| `loop.py`  | `adapt_loop` e i passi del ciclo: `run_su2`, `compute_metric`, `run_gmsh`, `write_log` |
| `core.py`  | La matematica: Hessiano, metrica Lp, limiti h_min/h_max/AR, complessità, scelta di D |
| `files.py` | Lettura e scrittura dei file: `flow.vtu`, `.su2`, `history.csv`, `.cfg`, `.geo`, `.pos` |
| `tests/`   | Controlli automatici (`cd adaptation && python3 -m pytest`) — non serve leggerli |

Serve solo numpy (per i test anche scipy e pytest).

## Cosa è stato corretto rispetto a `try_refinement.py`
1. **La complessità ora è quella richiesta.** Prima D veniva scelto *prima* di applicare
   h_max e l'aspect ratio, e questi la facevano salire (10 000 → 10 658 → 13 278 → 18 855).
   Ora D si cerca con una bisezione *dopo* i limiti.
2. **Il `.geo` adattato non è più frenato dalla baseline.** `Mesh.MeshSizeMin = hWall`,
   `Background Field`, ecc. vengono commentati e sostituiti con valori coerenti con
   `h_min`, `h_max`, `max_aspect_ratio`.
3. **`h_min = 1e-3` di default.** Senza limite, sull'urto l'Hessiano cresce ogni volta
   che la mesh si raffina e la mesh non smetterebbe mai di raffinarsi.
4. **Meno lavoro e meno file:** nessuna metrica dopo l'ultima soluzione; `metric.pos`
   (~10 MB) e restart non vengono tenuti (`keep_metric=True` per tenere i `.pos`).

## Aggiungere una metrica (es. goal-oriented)
In `loop.py`, funzione `compute_metric`, si aggiunge un caso:
```python
    if metric == "hessian":
        ...
    elif metric == "goal":
        M0 = ...          # la nuova metrica (non normalizzata)
```
La normalizzazione e i limiti vengono applicati dopo, uguali per tutte le metriche.
