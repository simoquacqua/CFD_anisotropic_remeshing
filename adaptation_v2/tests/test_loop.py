"""Test del ciclo completo con SU2 e Gmsh SIMULATI.

I finti eseguibili restituiscono i risultati reali gia' presenti in
Diagnostics/TestIterativeSteady/test-refinement/Iterations e verificano che
il ciclo scriva correttamente .cfg, .geo e metric.pos. Il test viene saltato
se quei dati non ci sono.
"""
import csv
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from adaptation_v2 import files
from adaptation_v2.loop import adapt_loop

REF = (Path(__file__).resolve().parents[2] /
       "Diagnostics/TestIterativeSteady/test-refinement")
ITERS = REF / "Iterations"
pytestmark = pytest.mark.skipif(not (ITERS / "iter_01" / "mesh.su2").is_file(),
                                reason="dati di riferimento assenti")

FAKE_SU2 = textwrap.dedent(f"""
    import shutil, sys
    from pathlib import Path
    cfg = Path(sys.argv[1]).read_text()
    assert "MESH_FILENAME= mesh.su2" in cfg
    assert "VOLUME_FILENAME= Results/flow" in cfg
    nelem = next(l for l in open("mesh.su2") if l.startswith("NELEM="))
    src = None
    for d in sorted(Path(r"{ITERS}").glob("iter_*")):
        if next(l for l in open(d / "mesh.su2") if l.startswith("NELEM=")) == nelem:
            src = d / "Results"
    assert src is not None, "mesh sconosciuta"
    assert "OUTPUT_FILES= ( RESTART, PARAVIEW, SURFACE_CSV )" in cfg
    for f in ("flow.vtu", "history.csv", "surface_flow.csv"):
        shutil.copy(src / f, Path("Results") / f)
    Path("Results/restart_flow.dat").write_text("restart finto")
    print("finto SU2 ok")
""")

FAKE_GMSH = textwrap.dedent(f"""
    import shutil, sys
    from pathlib import Path
    geo = Path(sys.argv[1]).read_text()
    out = sys.argv[sys.argv.index("-o") + 1]
    if sys.argv[1] == "adapted.geo":
        assert 'Merge "metric.pos";' in geo and Path("metric.pos").stat().st_size > 1000
        assert "// [adapt] disattivato: Mesh.MeshSizeMin = hWall;" in geo
        assert "// [adapt] disattivato: Background Field = 2;" in geo
        k = int(Path.cwd().name.split("_")[1])
        shutil.copy(Path(r"{ITERS}") / f"iter_{{k:02d}}" / "mesh.su2", out)
    else:
        shutil.copy(Path(r"{ITERS}") / "iter_00" / "mesh.su2", out)
    print("finto gmsh ok")
""")


FAKE_SU2_AD = textwrap.dedent("""
    import sys
    from pathlib import Path
    cfg = Path(sys.argv[1]).read_text()
    for opt in ("MATH_PROBLEM= DISCRETE_ADJOINT", "OBJECTIVE_FUNCTION= DRAG",
                "SOLUTION_FILENAME= Results/restart_flow.dat",
                "VOLUME_ADJ_FILENAME= Results/adjoint", "CONV_FILENAME= Results/history_adjoint"):
        assert opt in cfg, opt
    assert Path("Results/restart_flow.dat").is_file()
    # finto aggiunto: il flow.vtu con i campi rinominati (gli offset binari non cambiano)
    data = Path("Results/flow.vtu").read_bytes()
    for a, b in [(b'Name="Density"', b'Name="Adjoint_Density"'),
                 (b'Name="Momentum"', b'Name="Adjoint_Momentum"'),
                 (b'Name="Energy"', b'Name="Adjoint_Energy"')]:
        data = data.replace(a, b, 1)
    Path("Results/adjoint.vtu").write_bytes(data)
    print("finto aggiunto ok")
""")


@pytest.fixture
def fakes(tmp_path):
    su2 = tmp_path / "fake_su2.py"
    gmsh = tmp_path / "fake_gmsh.py"
    su2.write_text(FAKE_SU2)
    gmsh.write_text(FAKE_GMSH)
    su2_ad = tmp_path / "fake_su2_ad.py"
    su2_ad.write_text(FAKE_SU2_AD)
    return f"{sys.executable} {su2}", f"{sys.executable} {gmsh}", f"{sys.executable} {su2_ad}"


@pytest.mark.parametrize("initial", [True, False])
def test_full_loop(tmp_path, fakes, initial):
    su2_cmd, gmsh_cmd, _ = fakes
    work = tmp_path / "run"
    rows = adapt_loop(
        cfg_file=REF / "euler_steady.cfg", geo_file=REF / "diamond_euler.geo",
        initial_mesh=(REF / "diamond_euler.su2") if initial else None,
        work_dir=work, n_iter=3, complexity=[10000, 12000],
        h_min=None, h_max=0.3, max_aspect_ratio=10,
        su2_command=su2_cmd, gmsh_command=gmsh_cmd,
        intermediate_su2_options={"CONV_RESIDUAL_MINVAL": -6},
    )
    assert [r["elements"] for r in rows] == [3902, 26426, 31077]
    assert rows[-1]["CD"] == pytest.approx(0.0111304778)
    # controllo di qualita' della mesh nuova, solo nelle iterazioni con adattamento
    assert "edges_unit_pct" in rows[0] and "q_median" in rows[1] and "q_median" not in rows[2]
    assert rows[0]["mesh_complexity"] == pytest.approx(np.sqrt(3) / 4 * 26426, abs=0.1)
    # la complessita' ottenuta ora e' quella richiesta
    assert rows[0]["metric_complexity"] == pytest.approx(10000, rel=1e-3)
    assert rows[1]["metric_complexity"] == pytest.approx(12000, rel=1e-3)
    assert "metric_target" not in rows[2]
    # opzioni intermedie solo dove servono
    assert "CONV_RESIDUAL_MINVAL= -6" in (work / "iter_00/euler_steady.cfg").read_text()
    assert "CONV_RESIDUAL_MINVAL= -8" in (work / "iter_02/euler_steady.cfg").read_text()
    # metric.pos cancellato di default, log e config presenti
    assert not (work / "iter_01/metric.pos").exists()
    assert (work / "parametri.txt").is_file()
    with (work / "cycles_log.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 3


def test_refuses_to_overwrite(tmp_path, fakes):
    (tmp_path / "run" / "iter_00").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        adapt_loop(cfg_file=REF / "euler_steady.cfg", geo_file=REF / "diamond_euler.geo",
                   work_dir=tmp_path / "run", n_iter=1,
                   su2_command=fakes[0], gmsh_command=fakes[1])


def test_complexity_list_length_checked(tmp_path):
    with pytest.raises(ValueError):
        adapt_loop(cfg_file=REF / "euler_steady.cfg", geo_file=REF / "diamond_euler.geo",
                   work_dir=tmp_path / "run", n_iter=3, complexity=[1, 2, 3])


def test_vtu_reader_matches_mesh():
    flow = files.read_vtu(ITERS / "iter_00/Results/flow.vtu")
    n_elements, n_nodes = files.su2_mesh_counts(ITERS / "iter_00/mesh.su2")
    assert len(flow["points"]) == n_nodes
    assert len(flow["triangles"]) == n_elements == 3902
    assert files.get_field(flow, "Momentum_x").shape == (n_nodes,)


def test_full_loop_goal(tmp_path, fakes):
    su2_cmd, gmsh_cmd, su2_ad_cmd = fakes
    work = tmp_path / "goal"
    rows = adapt_loop(
        cfg_file=REF / "euler_steady.cfg", geo_file=REF / "diamond_euler.geo",
        initial_mesh=REF / "diamond_euler.su2", work_dir=work,
        n_iter=3, complexity=10000, metric="goal",
        h_min=1e-3, h_max=0.3, max_aspect_ratio=10,
        su2_command=su2_cmd, gmsh_command=gmsh_cmd, su2_adjoint_command=su2_ad_cmd,
    )
    assert [r["elements"] for r in rows] == [3902, 26426, 31077]
    # aggiunto lanciato solo prima di ogni adattamento, non all'ultima iterazione
    assert "adjoint_s" in rows[0] and "adjoint_s" in rows[1] and "adjoint_s" not in rows[2]
    assert not (work / "iter_02" / "adjoint.cfg").exists()
    assert rows[0]["metric_complexity"] == pytest.approx(10000, rel=1e-3)
    assert rows[1]["metric_complexity"] == pytest.approx(10000, rel=1e-3)
    assert (work / "iter_00" / "Results" / "adjoint.vtu").is_file()
    assert (work / "iter_00" / "Results" / "history.csv").is_file()   # non sovrascritto


def test_goal_requires_adjoint_program(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapt_loop(cfg_file=REF / "euler_steady.cfg", geo_file=REF / "diamond_euler.geo",
                   work_dir=tmp_path / "x", n_iter=2, metric="goal",
                   su2_command=sys.executable, gmsh_command=sys.executable,
                   su2_adjoint_command="programma_che_non_esiste")


def test_read_su2_mesh_matches_vtu():
    pts, tri = files.read_su2_mesh(ITERS / "iter_00/mesh.su2")
    flow = files.read_vtu(ITERS / "iter_00/Results/flow.vtu")
    assert len(tri) == 3902 and len(pts) == len(flow["points"])
