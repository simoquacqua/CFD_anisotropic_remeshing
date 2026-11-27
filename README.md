# Anisotropic mesh adaptation for a supersonic diamond airfoil (SU2 + Gmsh)

This folder contains a metric-based anisotropic mesh adaptation loop for
steady inviscid flow, built around **SU2 8.5.0** (flow solver and discrete
adjoint) and **Gmsh 4.13.1** (BAMG anisotropic remesher), with all metric
computations written in Python/NumPy.

**Test case:** diamond (double-wedge) airfoil, chord *c* = 1, thickness
*t/c* = 0.07 (half-angle 4.0°), Euler equations, *M*∞ = 2, α = 0. The farfield is a
circle of radius 5*c*. The reference drag coefficient from exact shock-expansion theory is
**C_D = 0.011341**.

Two adaptation strategies are implemented:

| Metric | What it controls | Formula |
|---|---|---|
| `hessian` | interpolation error of a scalar field (default: pressure), *L^p* norm | M = D · det\|H\|^(−1/(2p+2)) · \|H\| |
| `goal` | error on the drag coefficient C_D (Loseille–Dervieux–Alauzet 2010) | H_go = Σ_j \|∂λ_j/∂x\| \|H(F_j)\| + \|∂λ_j/∂y\| \|H(G_j)\|, M = D · det(H_go)^(−1/4) · H_go |

Here *H* is the nodal Hessian (least-squares reconstruction), *F_j* and *G_j* are the Euler
fluxes, and λ is the discrete adjoint of C_D from SU2. *D* is found by bisection so that the
metric complexity C = ∫ √det M matches the requested value; the resulting mesh has
N_triangles ≈ 2.3 C.

---

## Requirements

- SU2 8.5.0 with `SU2_CFD` and `SU2_CFD_AD` (the AD build is needed for the
  goal-oriented metric). Building with MPI is optional.
- Gmsh 4.13.1, with BAMG (`Mesh.Algorithm = 7`).
- Python ≥ 3.9 with `numpy`. `pytest` is only needed for the tests, and
  `matplotlib` only for `convergence.py`.

All commands below are run **from this folder**, inside the Docker container that
provides SU2 and Gmsh.

---

## Folder layout

```
.
├── run_adapt_v2.py        main entry point (command-line driver of the loop)
├── adaptation_v2/         the adaptation package
│   ├── core.py            metrics, Hessian/gradient reconstruction, normalization, corner refinement, quality
│   ├── files.py           readers/writers: SU2 mesh, VTU, history, .cfg editing, Gmsh .geo and metric .pos
│   ├── loop.py            the adaptation loop (SU2 → adjoint → metric → Gmsh → next mesh)
│   └── tests/             unit tests (pytest)
├── cases/<case>/          input of each case: one .cfg (SU2), one .geo (Gmsh), optional initial .su2 mesh
├── runs/<case>/<run>/     results, created by the loop (one subfolder per adaptation iteration)
├── cd_breakdown.py        splits the C_D error by zone of the airfoil surface
├── adjoint_check.py       writes the rescaled adjoint and the goal-oriented error indicator for ParaView
├── convergence.py         C_D convergence vs. mesh size over several runs
├── solver_test.sh         compares solver settings on one fixed mesh
├── run_adapt.py           first version of the driver (uses adaptation/)
└── adaptation/            first version of the package (kept for reference)
```

---

## Quick start

```bash
# Hessian-based adaptation, 10 SU2 solutions, complexity 20,000
python3 run_adapt_v2.py --case diamond_euler --run hessian_20k --metric hessian \
    --complexity 20000 --n-iter 10

# Goal-oriented adaptation on C_D with refinement at the airfoil corners
python3 run_adapt_v2.py --case diamond_euler --run goal_20k_hc1e-4 --metric goal \
    --complexity 20000 --n-iter 10 --h-corner 1e-4

# Same, in parallel (SU2 must be built with MPI)
python3 run_adapt_v2.py ... --su2 "mpirun -n 4 SU2_CFD" --su2-adjoint "mpirun -n 4 SU2_CFD_AD"

python3 run_adapt_v2.py --help      # all options
```

The loop refuses to overwrite an existing run folder unless `--overwrite` is given.
Several loops can run at the same time if each one has a different `--run` name.

### Main options

| Option | Default | Meaning |
|---|---|---|
| `--case` | `diamond_euler` | folder in `cases/` |
| `--run` | `prova` | name of the results folder in `runs/<case>/` |
| `--metric` | `hessian` | `hessian` or `goal` |
| `--n-iter` | 2 | number of SU2 solutions (adapted meshes = n_iter − 1) |
| `--complexity` | 5000 | target complexity: one value, or one value per adapted mesh |
| `--field`, `--p` | `Pressure`, 2 | field and *L^p* norm of the Hessian metric (the goal metric always uses p = 1) |
| `--h-min`, `--h-max` | 1e-3, 0.3 | minimum and maximum element size |
| `--max-ar` | 10 | maximum element aspect ratio |
| `--h-corner` | off | isotropic size imposed at the four airfoil corners, growing as h = h_c + 0.3·d |
| `--corner-ratio` | 3 | the corner size is reduced by at most this factor per iteration (prevents mesh blow-up) |
| `--corner-start` | 1 | first iteration at which the corner size is imposed |
| `--adjoint-cfl` | 5 | CFL of the adjoint; halved automatically (twice at most) if the adjoint diverges |
| `--intermediate-residual` | −6 | residual target for SU2 in all iterations except the last (the last one uses the .cfg value) |
| `--su2`, `--su2-adjoint` | `SU2_CFD`, `SU2_CFD_AD` | solver commands, e.g. with `mpirun` |

---

## What one iteration does

1. **Flow solution** (`SU2_CFD`): Roe or JST scheme with an implicit Euler pseudo-time step and an
   adaptive CFL. The starting point is always the freestream.
2. **Adjoint of C_D** (`SU2_CFD_AD`, goal metric only), restarted from the flow solution. SU2
   returns the fixed-point adjoint Ū = Pᵀλ, where P = |V|/Δt + Ã. The loop rescales it as
   λ_i ≈ CFL · Ū_i / s_i, where s_i is the sum of spectral radii over the faces of the dual cell.
   This removes the dependence on cell size.
3. **Metric**:
   1. Hessian or goal-oriented raw metric;
   2. optional isotropic corner refinement, intersected with the metric, eigenvalue by
      eigenvalue: max(λ, 1/h(d)²);
   3. clipping to h_min, h_max and AR_max;
   4. scaling by D to the requested complexity. The corner refinement counts towards the
      complexity.
4. **New mesh**: the metric is written as a Gmsh `.pos` background field and BAMG
   generates the next mesh.
5. **Quality report** of the new mesh with respect to the metric (UGAWG conventions):
   - fraction of edges with metric length in [√2/2, √2];
   - element quality Q (1 = equilateral in the metric);
   - achieved vs. requested complexity.

Each run writes:

```
runs/<case>/<run>/
├── parametri.txt          all parameters of the run
├── cycles_log.csv         one row per iteration: elements, nodes, C_D, C_L, SU2 iterations, residuals,
│                          adjoint residual and CFL, metric scaling D, corner nodes, mesh quality, timings
└── iter_XX/
    ├── mesh.su2           mesh of this iteration
    ├── su2.log, adjoint.log
    └── Results/           flow.vtu, adjoint.vtu, surface_flow.csv, history.csv, restart files
```

---

## Post-processing scripts

| Script | Usage | Output |
|---|---|---|
| `cd_breakdown.py` | `python3 cd_breakdown.py runs/<case>/<run>` | For each iteration, integrates the surface pressure and compares it, face by face, with shock-expansion theory. Prints the C_D error contributed by the leading edge, front faces, shoulder, rear faces and trailing edge. |
| `adjoint_check.py` | `python3 adjoint_check.py runs/<case>/<run>/iter_XX --cfl <adjoint CFL>` | Writes `Results/adjoint_check.vtu` with fields `lambda_*`, `grad_lambda_rho` and `error_indicator` (= tr H_go, the quantity that drives the goal metric). Also contains the raw `Ubar_rho` for comparison. |
| `convergence.py` | `python3 convergence.py --help` | Over the last K iterations of every run of a case: mean, min and max C_D vs. elements, and error with respect to 0.011341 (`convergence_summary.csv`, `convergence.png`). |
| `solver_test.sh` | `bash solver_test.sh runs/<case>/<run>/iter_XX` | Runs several solver settings on one fixed mesh and prints iterations, time, residual, C_D and C_L. |

---

## Tests

```bash
python3 -m pytest -q adaptation_v2/tests
```

The tests cover the Hessian and gradient reconstruction on analytic fields, metric
normalization and complexity, corner refinement, the quality measures, and file I/O. Tests that
need Gmsh or SU2 are skipped when the programs are not available.

---

## Main findings so far (α = 0)

- **Error source.** The C_D error is dominated by the singular trailing-edge node. Its
  contribution decreases only as O(h_TE): about 0.046·h_TE for the node alone. The goal-oriented
  error indicator scales there as h⁻³.
- **Corner refinement.** It removes most of that error:

  | h_c | C_D error |
  |---|---|
  | 1e-3 | ≈ −1.07 % |
  | 3e-4 | −0.42 % |
  | 1e-4 | −0.16 % (C_D = 0.011323, identical at complexity 10,000 and 20,000) |

  The observed order is ≈ 0.88, and Richardson extrapolation gives −0.03 %.
- **Element placement.** At the same complexity (20,000), the goal-oriented metric puts
  57–63 % of the elements inside the region that influences the wall pressure (bounded by the
  leading-edge shocks and by the Mach lines from the trailing edge). The Hessian metric puts
  3–5 % there, because it refines shocks and fans out to the farfield.
- **Adjoint structure.** The refined bands of the goal-oriented mesh follow the discontinuities
  of the drag adjoint along the Mach lines from the corners. This agrees with the analytic
  adjoint of Lozano & Ponsin (*Aerospace* 12(6):494, 2025).
- **Mesh quality.** About 97–99 % of the edges have metric length in [√2/2, √2]. The median
  quality is Q ≈ 0.96–0.97.

## References

- A. Loseille, A. Dervieux, F. Alauzet, *Fully anisotropic goal-oriented mesh adaptation for 3D
  steady Euler equations*, J. Comput. Phys. 229 (2010).
- C. Lozano, J. Ponsin, *On the characteristic structure of the adjoint Euler equations and the
  analytic adjoint solution of supersonic inviscid flows*, Aerospace 12(6):494 (2025).
- M. B. Giles, N. A. Pierce, *Adjoint equations in CFD: duality, boundary conditions and solution
  behaviour*, AIAA 97-1850 (1997).