# -*- coding: utf-8 -*-

"""FFT grids, and fragments registered on a parent cell's grid.

An atom's energy in a plane-wave calculation depends slightly on where it sits
relative to the FFT grid (the "egg-box" effect: 2-3 meV/Å in the forces for a
fractional shift, 0.06 meV/Å for a whole grid step). Many-body increments
difference calculations of fragments against each other and against their
parent cell, so the egg-box error cancels only if every atom keeps its offset
from the grid:

* the parent cell's grid is set explicitly, NG points along each axis with a
  spacing h <= ``max_spacing``;
* each fragment's box is a whole number n of those steps, n·h, with NGX = n
  (NGXF = 2n) set explicitly;
* each fragment is the parent's coordinates shifted by whole grid steps.

Grid sizes are even and 2·3·5·7-smooth, as VASP's FFTs prefer. This follows
the MBE prototype's ``gen_registered.py``, and reproduces it.

Boxes are cubes (in grid steps) sized for the fragment's largest extent plus a
padding, at least 12 Å: 12 Å was within 2 meV/Å of 16 Å for a water dimer
increment at a third of the cost. Padding matters for extended fragments,
which interact with their periodic images in an affordable box (extent + 15 Å
roughly halves far-pair errors at ~6× the cost, and the dipole correction does
not cure it), so only compact fragments belong in a periodic code.
"""

import math

import numpy as np

#: The prototype's grid spacing (Å): 12.4297 Å / 150
DEFAULT_MAX_SPACING = 0.0829
DEFAULT_PADDING = 7.5
MINIMUM_BOX = 12.0


def good(n):
    """Whether n is even and has no prime factors but 2, 3, 5 and 7."""
    if n <= 0 or n % 2:
        return False
    m = n
    for p in (2, 3, 5, 7):
        while m % p == 0:
            m //= p
    return m == 1


def next_good(n):
    """The smallest good n' >= n."""
    n = max(2, int(n))
    while not good(n):
        n += 1
    return n


def grid_points(length, max_spacing=DEFAULT_MAX_SPACING):
    """Grid points along an axis of ``length`` Å with spacing <= ``max_spacing``."""
    return next_good(math.ceil(length / max_spacing))


def cell_grid(cell, max_spacing=DEFAULT_MAX_SPACING):
    """NG along each lattice vector of a cell (3x3, vectors as rows, Å)."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    return [grid_points(length, max_spacing) for length in lengths]


def _orthorhombic(cell):
    cell = np.asarray(cell, dtype=float)
    off = cell - np.diag(np.diag(cell))
    if np.abs(off).max() > 1e-8 * np.abs(cell).max():
        raise ValueError(
            "Fragments can be registered only on an orthorhombic cell (lattice "
            "vectors along x, y and z)."
        )
    return np.diag(cell)


def register(
    coordinates,
    reference_cell,
    max_spacing=DEFAULT_MAX_SPACING,
    padding=DEFAULT_PADDING,
    minimum=MINIMUM_BOX,
):
    """The box, grid and coordinates of a fragment registered on a parent cell.

    Parameters
    ----------
    coordinates : array-like
        (n, 3) Cartesian coordinates (Å) in the parent cell's frame.
    reference_cell : array-like
        The parent cell, 3x3, orthorhombic.
    max_spacing : float
        The parent grid's largest spacing (Å); its NG is :func:`cell_grid`.
    padding : float
        Added to the fragment's largest extent (Å).
    minimum : float
        The smallest box edge (Å).

    Returns
    -------
    dict
        "ng": [n, n, n] grid points of the box (NGXF = 2n),
        "box": (3,) box edges in Å (n·h along each axis),
        "coordinates": (n, 3) the fragment shifted by whole grid steps so it is
        centred in the box,
        "h": (3,) the grid steps, and "reference_ng": the parent's NG.
    """
    xyz = np.asarray(coordinates, dtype=float)
    edges = _orthorhombic(reference_cell)
    reference_ng = cell_grid(np.diag(edges), max_spacing)
    h = edges / np.array(reference_ng)
    target = max(minimum, float(np.ptp(xyz, axis=0).max()) + padding)
    ng = [next_good(math.ceil(target / step)) for step in h]
    box = np.array(ng) * h
    centre = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    k = np.round((box / 2 - centre) / h)
    return {
        "ng": ng,
        "box": box,
        "coordinates": xyz + k * h,
        "h": h,
        "reference_ng": reference_ng,
    }


def register_cell(coordinates, cell, max_spacing=DEFAULT_MAX_SPACING):
    """A cell on its own explicit grid, its atoms shifted by whole grid steps
    to centre them in the cell -- as the prototype wrote the cell, so that the
    cell and its fragments share the same atom-to-grid offsets.

    Returns the same keys as :func:`register`.
    """
    xyz = np.asarray(coordinates, dtype=float)
    edges = _orthorhombic(cell)
    ng = cell_grid(np.diag(edges), max_spacing)
    h = edges / np.array(ng)
    centre = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    k = np.round((edges / 2 - centre) / h)
    return {
        "ng": ng,
        "box": edges,
        "coordinates": xyz + k * h,
        "h": h,
        "reference_ng": ng,
    }
