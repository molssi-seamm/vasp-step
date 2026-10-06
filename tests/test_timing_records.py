# -*- coding: utf-8 -*-
"""The timing records VASP runs write (seamm_exec campaign 2026-10-05)."""

import gzip
from pathlib import Path
from types import SimpleNamespace

from vasp_step.energy import timing_descriptors

OUTCAR = Path(__file__).parent / "data" / "pilot" / "r2_d00_18" / "OUTCAR.gz"

INCAR = """\
SYSTEM = test
PREC = Accurate
ENCUT = 1200
EDIFF = 1e-7
ALGO = All
ISPIN = 1
IBRION = -1
NSW = 0
LDIPOL = .TRUE.
"""
KPOINTS = "Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n"
POSCAR = "EC\n1.0\n15 0 0\n0 15 0\n0 0 15\nC H O\n3 4 3\nCartesian\n"
POTCAR = (
    "  PAW_PBE C_h\n   ZVAL   =    4.000\n PAW_PBE H_h\n   ZVAL   =    1.000\n"
    " PAW_PBE O_h\n   ZVAL   =    6.000\n"
)


def test_descriptors():
    conf = SimpleNamespace(
        atoms=SimpleNamespace(atomic_numbers=[6, 6, 6, 1, 1, 1, 1, 8, 8, 8]),
        charge=0,
        spin_multiplicity=1,
        periodicity=3,
        volume=3375.0,
    )
    files = {"INCAR": INCAR, "KPOINTS": KPOINTS, "POSCAR": POSCAR, "POTCAR": POTCAR}
    with gzip.open(OUTCAR, "rt", errors="replace") as fd:
        outcar = fd.read()
    d = timing_descriptors(
        files, outcar, conf, model="r2SCAN / r2SCAN", potentials="C_h H_h O_h"
    )
    assert d["encut"] == 1200.0 and d["ediff"] == 1e-7
    assert d["algo"] == "All" and d["prec"] == "Accurate" and d["ispin"] == "1"
    assert d["task"] == "energy"
    assert d["kpoints"] == 1
    assert d["n_atoms"] == 10 and d["n_heavy"] == 6 and d["volume"] == 3375.0
    assert abs(d["grid"] - 3375.0 * (1200.0 / 500.0) ** 1.5) < 1e-6
    assert d["nelect"] == 3 * 4 + 4 * 1 + 3 * 6
    assert d["mpi_ranks"] == 16
    assert d["electronic_steps"] == 21 and d["ionic_steps"] == 1
    assert abs(d["code_seconds"] - 341.7) < 1e-6
    assert abs(d["cpu_seconds"] - 335.824) < 1e-6
    assert d["terminated_normally"] is True
    assert d["model"] == "r2SCAN / r2SCAN"


def test_task_kinds():
    assert (
        timing_descriptors({"INCAR": "IBRION = 2\nNSW = 50\n"}, None)["task"] == "opt"
    )
    assert (
        timing_descriptors({"INCAR": "IBRION = 6\nNSW = 1\n"}, None)["task"] == "force"
    )
    assert timing_descriptors({"INCAR": ""}, None)["task"] == "energy"
    assert timing_descriptors({}, None)["task"] == "energy"


def test_timing_spec():
    from vasp_step import energy

    assert energy.TIMING_SPEC["size"] == ["nelect", "grid"]
    assert energy.TIMING_SPEC["multiplier"] == "kpoints"
