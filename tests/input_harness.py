"""Generate the Energy substep's VASP inputs without a flowchart.

Used by the input-identity tests: the inputs for a few representative settings
were captured from vasp_step before the input builder was factored out
(tests/data/inputs/<case>/), and must stay byte-identical. The POTCAR library
is fake (licensed files cannot be shipped): each "POTCAR" is a short text
naming the potential.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA = Path(__file__).parent / "data"
POTCARS = DATA / "potcars"

#: The fake potential library: set -> name -> {"file", "Emax"}
ENMAX = {"H": 250.0, "O": 400.0, "Si": 245.3, "Li_sv": 499.0, "F": 400.0}


def potential_metadata():
    data = {"potpaw_PBE.64": {}}
    for name, emax in ENMAX.items():
        path = POTCARS / f"{name}.POTCAR"
        data["potpaw_PBE.64"][name] = {"file": str(path), "Emax": emax}
    return data


def write_fake_potcars():
    POTCARS.mkdir(parents=True, exist_ok=True)
    for name in ENMAX:
        (POTCARS / f"{name}.POTCAR").write_text(
            f"  PAW_PBE {name} (fake for tests)\n  ENMAX = {ENMAX[name]:.3f}\n"
            " End of Dataset\n"
        )


def water_cell(db):
    """Two waters in a 6 Å cubic cell."""
    system = db.create_system(name="water")
    configuration = system.create_configuration(periodicity=3, name="two")
    configuration.cell.parameters = [6.0, 6.0, 6.0, 90.0, 90.0, 90.0]
    xyz = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.957, 1.0, 1.0],
            [0.76, 1.927, 1.0],
            [3.9, 1.2, 1.1],
            [4.4, 2.0, 1.2],
            [4.5, 0.5, 1.3],
        ]
    )
    configuration.coordinate_system = "Cartesian"
    configuration.atoms.append(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], symbol=["O", "H", "H"] * 2
    )
    return system, configuration


def silicon(db):
    """Diamond silicon, primitive cell."""
    system = db.create_system(name="")
    configuration = system.create_configuration(periodicity=3, name="")
    configuration.cell.parameters = [3.8669, 3.8669, 3.8669, 60.0, 60.0, 60.0]
    configuration.coordinate_system = "fractional"
    configuration.atoms.append(
        x=[0.0, 0.25], y=[0.0, 0.25], z=[0.0, 0.25], symbol=["Si", "Si"]
    )
    return system, configuration


def lif(db):
    """LiF rock salt (conventional cell), Li first in the input."""
    system = db.create_system(name="LiF")
    configuration = system.create_configuration(periodicity=3, name="rocksalt")
    configuration.cell.parameters = [4.03, 4.03, 4.03, 90.0, 90.0, 90.0]
    configuration.coordinate_system = "fractional"
    li = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    f = [[0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]]
    xyz = np.array(li + f, dtype=float)
    configuration.atoms.append(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], symbol=["Li"] * 4 + ["F"] * 4
    )
    return system, configuration


#: case name -> (structure builder, parameter values)
CASES = {
    "water_gamma": (
        water_cell,
        {
            "potentials": {"O": "O", "H": "H"},
            "k-grid method": "𝚪-point",
            "plane-wave cutoff": 700.0,
            "calculate stress": "yes",
        },
    ),
    "si_spacing": (
        silicon,
        {
            "potentials": {"Si": "Si"},
            "k-grid method": "grid spacing",
            "odd grid": "yes",
            "centering": "Monkhorst-Pack",
            "calculate stress": "only pressure",
            "occupation type": "the Methfessel-Paxton method",
        },
    ),
    "lif_explicit": (
        lif,
        {
            "potentials": {"Li": "Li_sv", "F": "F"},
            "k-grid method": "explicit grid dimensions",
            "na": 3,
            "nb": 3,
            "nc": 3,
            "spin polarization": "collinear",
            "calculate stress": "no",
            "occupation type": (
                "the tetrahedron method with Blöchl corrections with Fermi-Dirac"
                " smearing"
            ),
            "extra keywords": ["NBANDS=40", "LCHARG=.False."],
        },
    ),
}


#: Cases run through the Optimization substep (its own keywords on top)
OPTIMIZATION = {"water_optimization": "water_gamma"}


def generate(case, db, tmp_path):
    """The substep's {INCAR, POTCAR, KPOINTS, POSCAR} for a case."""
    import vasp_step

    if case in OPTIMIZATION:
        builder, values = CASES[OPTIMIZATION[case]]
        energy = vasp_step.Optimization()
    else:
        builder, values = CASES[case]
        energy = vasp_step.Energy()
    system, configuration = builder(db)
    for key, value in values.items():
        energy.parameters[key].value = value
    energy._id = (1, 1)
    energy._timing_data = None
    energy.get_system_configuration = lambda *a, **k: (system, configuration)
    energy.parent = SimpleNamespace(
        potential_metadata=potential_metadata(), get_value=lambda v: v
    )
    P = energy.parameters.current_values_to_dict(context={})
    return energy.get_input(P)


def capture(directory, db, tmp_path):
    """Write every case's inputs under ``directory/<case>/``."""
    for case in list(CASES) + list(OPTIMIZATION):
        files = generate(case, db, tmp_path)
        out = Path(directory) / case
        out.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (out / name).write_text(text)
    (Path(directory) / "README.md").write_text(
        "Inputs of vasp_step's Energy substep captured before the input builder "
        "was factored out (vasp_step 2026.9.29 + dev 1690fe2). They must stay "
        "byte-identical. Regenerate only for an intended change of the inputs.\n"
    )
    return json.dumps(sorted(CASES))
