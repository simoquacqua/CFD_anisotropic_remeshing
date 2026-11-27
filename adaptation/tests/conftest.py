"""Mesh sintetiche e campi analitici per i test (richiede scipy)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

# rende importabile "core" lanciando pytest dalla cartella adaptation/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # "import core"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # "import adaptation"


@pytest.fixture
def square_mesh():
    """~3000 nodi casuali in [-1,1]^2 + i 4 vertici, triangolati con Delaunay."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(3000, 2))
    pts = np.vstack([pts, [[-1, -1], [1, -1], [1, 1], [-1, 1]]])
    tri = Delaunay(pts).simplices
    return pts, tri


def rotated(points, angle_deg):
    """Coordinate (x', y') ruotate di angle_deg."""
    a = np.deg2rad(angle_deg)
    x, y = points[:, 0], points[:, 1]
    return np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y


def quadratic_field(points, angle_deg=35.0):
    """p = 100 x'^2 + y'^2 e il suo Hessiano esatto (costante)."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    xr, yr = rotated(points, angle_deg)
    H = 2 * np.array([[100 * c * c + s * s, 99 * c * s],
                      [99 * c * s, 100 * s * s + c * c]])
    return 100 * xr**2 + yr**2, H


def shock_like_field(points, angle_deg=35.0, width=0.05):
    """Salto liscio tanh lungo x': piatto quasi ovunque, ripido su una linea.
    Imita un urto obliquo: e' il caso in cui il clip con h_max conta."""
    xr, _ = rotated(points, angle_deg)
    return np.tanh(xr / width)
