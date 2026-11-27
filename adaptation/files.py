"""
files.py - Leggere e scrivere i file di SU2 e Gmsh.

(Il nome e' files.py e non io.py perche' "io" e' gia' un modulo di Python.)

Funzioni:
    read_vtu           legge flow.vtu di SU2 -> punti, triangoli, campi (Pressure, ...)
    get_field          prende un campo dal risultato di read_vtu
    su2_mesh_counts    numero di elementi e nodi di una mesh .su2
    read_history       ultima riga di history.csv (CD, CL, residui, ...)
    set_cfg_options    cambia delle opzioni in un file .cfg di SU2
    make_adapted_geo   prepara il .geo che usa la metrica
    write_metric_pos   scrive la metrica nel formato .pos di Gmsh
"""
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# =============================================================================
# Lettura di flow.vtu
# =============================================================================
#
# SU2 scrive flow.vtu cosi':
#   - una parte di testo XML che dice quali array ci sono e dove iniziano;
#   - dopo la riga <AppendedData encoding="raw"> e il carattere "_",
#     tutti i dati in binario, uno dopo l'altro.
# Ogni array binario inizia con 8 byte che dicono quanto e' lungo.

VTK_TYPES = {"Float32": np.float32, "Float64": np.float64,
             "Int32": np.int32, "Int64": np.int64, "UInt8": np.uint8}
VTK_TRIANGLE = 5
VTK_QUAD = 9


def read_vtu(path):
    """Legge flow.vtu e restituisce un dizionario con:
        'points'     (n, 2) coordinate
        'triangles'  (m, 3) triangoli (i quadrilateri vengono spezzati in due)
        'fields'     dizionario {nome: array}, es. fields['Pressure']
    """
    data = Path(path).read_bytes()

    # --- 1. separa la parte di testo da quella binaria -----------------------
    tag = b'<AppendedData encoding="raw">'
    tag_position = data.find(tag)
    if tag_position < 0:
        raise ValueError(
            f"{path}: formato non riconosciuto (atteso VTU binario di SU2)")
    binary_start = data.index(b"_", tag_position) + 1

    xml_text = data[:tag_position].decode("ascii") + "</VTKFile>"
    root = ET.fromstring(xml_text)
    if root.get("header_type") != "UInt64":
        raise ValueError(
            f"{path}: header_type inatteso {root.get('header_type')}")

    # --- 2. funzione che legge un array binario ------------------------------
    def read_array(xml_element):
        start = binary_start + int(xml_element.get("offset"))
        n_bytes = int(np.frombuffer(data, np.uint64, count=1, offset=start)[0])
        dtype = np.dtype(VTK_TYPES[xml_element.get("type")])
        values = np.frombuffer(data, dtype, count=n_bytes //
                               dtype.itemsize, offset=start + 8)
        n_components = int(xml_element.get("NumberOfComponents", "1"))
        if n_components > 1:
            values = values.reshape(-1, n_components)
        return values

    piece = root.find("UnstructuredGrid/Piece")

    # --- 3. punti ------------------------------------------------------------
    points = read_array(piece.find("Points/DataArray")).astype(float)
    points = points[:, :2].copy()                 # teniamo solo x, y

    # --- 4. celle -> triangoli -----------------------------------------------
    cell_arrays = {}
    for element in piece.find("Cells"):
        cell_arrays[element.get("Name")] = read_array(element)
    connectivity = cell_arrays["connectivity"]
    offsets = cell_arrays["offsets"]              # dove finisce ogni cella
    types = cell_arrays["types"]

    triangles = []
    start = 0
    for k in range(len(types)):
        nodes = connectivity[start:offsets[k]]
        if types[k] == VTK_TRIANGLE:
            triangles.append(nodes)
        elif types[k] == VTK_QUAD:                # quadrilatero a,b,c,d -> due triangoli
            a, b, c, d = nodes
            triangles.append([a, b, c])
            triangles.append([a, c, d])
        start = offsets[k]
    triangles = np.array(triangles, dtype=np.int64)

    # --- 5. campi ai nodi ----------------------------------------------------
    fields = {}
    for element in piece.find("PointData"):
        fields[element.get("Name")] = read_array(element).astype(float)

    return {"points": points, "triangles": triangles, "fields": fields}


def get_field(flow, name):
    """Campo scalare per nome. Per i vettori si usa il suffisso: 'Momentum_x', 'Velocity_y'."""
    fields = flow["fields"]
    if name in fields:
        return fields[name]
    if name.endswith("_x") or name.endswith("_y"):
        base = name[:-2]
        component = 0 if name.endswith("_x") else 1
        if base in fields:
            return fields[base][:, component]
    raise KeyError(f"Campo '{name}' non trovato. Disponibili: {list(fields)}")


# =============================================================================
# File di SU2: mesh, history, cfg
# =============================================================================

def su2_mesh_counts(path):
    """Restituisce (numero_elementi, numero_nodi) leggendo NELEM= e NPOIN=."""
    n_elements = None
    with open(path) as f:
        for line in f:
            if line.startswith("NELEM="):
                n_elements = int(line.split("=")[1])
            elif line.startswith("NPOIN="):
                n_nodes = int(line.split("=")[1].split()[0])
                return n_elements, n_nodes
    raise ValueError(f"{path}: NELEM/NPOIN non trovati")


def read_history(path):
    """Ultima riga di history.csv come dizionario, es. {'CD': 0.0111, 'CL': ..., ...}."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = [name.strip().strip('"')
             for name in rows[0]]   # SU2 mette virgolette e spazi
    last_row = rows[-1]
    result = {}
    for name, value in zip(names, last_row):
        result[name] = float(value)
    return result


def set_cfg_options(cfg_text, options):
    """Cambia le opzioni di un .cfg di SU2.

    cfg_text: testo del file .cfg
    options:  dizionario {"OPZIONE": valore}
    Se l'opzione c'e' gia' viene sostituita, altrimenti viene aggiunta in fondo.
    """
    new_lines = []
    already_set = set()
    for line in cfg_text.splitlines():
        key = line.split("=")[0].strip() if "=" in line else ""
        if key in options and not line.lstrip().startswith("%"):
            if key in already_set:
                # opzione ripetuta
                new_lines.append("% [adapt] duplicato: " + line)
            else:
                new_lines.append(f"{key}= {options[key]}")
                already_set.add(key)
        else:
            new_lines.append(line)

    missing = [key for key in options if key not in already_set]
    if missing:
        new_lines.append("")
        new_lines.append("% ===== opzioni aggiunte da adaptation =====")
        for key in missing:
            new_lines.append(f"{key}= {options[key]}")
    return "\n".join(new_lines) + "\n"


# =============================================================================
# File di Gmsh: .geo e .pos
# =============================================================================

# Opzioni del .geo di partenza che darebbero fastidio alla metrica
# (es. Mesh.MeshSizeMin = hWall impedirebbe di raffinare sotto hWall sugli urti)
GEO_OPTIONS_TO_DISABLE = [
    "Background Field", "Background Mesh", "Merge",
    "Mesh.MeshSize", "Mesh.CharacteristicLength",
    "Mesh.Algorithm", "Mesh.AnisoMax", "Mesh.SmoothRatio",
]


def make_adapted_geo(geo_text, metric_file="metric.pos", h_min=None, h_max=None, aniso_max=None):
    """Prende il .geo originale e restituisce il testo del .geo adattato:
    commenta le opzioni di taglia originali e aggiunge in fondo la metrica."""
    lines = []
    for line in geo_text.splitlines():
        if line.strip().startswith(tuple(GEO_OPTIONS_TO_DISABLE)):
            lines.append("// [adapt] disattivato: " + line.strip())
        else:
            lines.append(line)

    lines.append("")
    lines.append(
        "// ===== ADATTAMENTO ANISOTROPO (aggiunto da adaptation) =====")
    lines.append(f'Merge "{metric_file}";')
    lines.append("Background Mesh View[0];")
    lines.append("Mesh.MeshSizeFromPoints = 0;")
    lines.append("Mesh.MeshSizeFromCurvature = 0;")
    lines.append("Mesh.MeshSizeExtendFromBoundary = 0;")
    lines.append(
        "Mesh.Algorithm = 7;   // BAMG: l'algoritmo 2D di Gmsh che segue metriche anisotrope")
    lines.append("Mesh.SmoothRatio = 3;")
    if h_min is not None:
        lines.append(f"Mesh.MeshSizeMin = {h_min};")
    if h_max is not None:
        lines.append(f"Mesh.MeshSizeMax = {h_max};")
    if aniso_max is not None:
        lines.append(f"Mesh.AnisoMax = {aniso_max};")
    return "\n".join(lines) + "\n"


def write_metric_pos(path, points, triangles, M):
    """Scrive la metrica come vista di Gmsh (.pos).

    Per ogni triangolo una riga:
        TT(x1,y1,z1, x2,y2,z2, x3,y3,z3){ tensore 3x3 nodo 1, nodo 2, nodo 3 };
    Il tensore 3x3 e' [[Mxx, Mxy, 0], [Mxy, Myy, 0], [0, 0, 1]]
    (la parte z non conta in 2D, ma deve esserci).
    """
    if not np.all(np.isfinite(M)):
        raise ValueError("La metrica contiene NaN o infiniti")

    with open(path, "w") as f:
        f.write('View "metric" {\n')
        for a, b, c in triangles:
            coords = []
            tensors = []
            for node in (a, b, c):
                x, y = points[node]
                coords += [x, y, 0.0]
                Mxx, Mxy, Myy = M[node, 0, 0], M[node, 0, 1], M[node, 1, 1]
                tensors += [Mxx, Mxy, 0.0,
                            Mxy, Myy, 0.0,
                            0.0, 0.0, 1.0]
            coords_text = ",".join(f"{v:.10g}" for v in coords)
            tensors_text = ",".join(f"{v:.10g}" for v in tensors)
            f.write(f"TT({coords_text}){{{tensors_text}}};\n")
        f.write("};\n")
