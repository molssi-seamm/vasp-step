"""The batch path (get_task/analyze_task) against the MBE prototype's own VASP
runs of the pilot frame (tests/data/pilot/README.md)."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import seamm_exec
from seamm_exec.evaluator import AnalysisError
from seamm_util import Q_

from vasp_step import batch, grid, resolver

from . import input_harness as harness

PILOT = Path(__file__).parent / "data" / "pilot"
EV_KJ = Q_(1.0, "eV").m_as("kJ/mol")
FRAGS = json.loads((PILOT / "frags_subset.json").read_text())
BOX = FRAGS["box"]
CELL = np.eye(3) * BOX
X = np.array(FRAGS["X"])
EXPECTED = json.loads((PILOT / "expected.json").read_text())
MC = {
    "level": "VASP:DFT@r2SCAN-D4/PAW-hard@1200",
    "method": "r2SCAN-D4",
    "basis": "PAW-hard",
    "cutoff": "1200",
    "step": "VASP",
}


def prototype_register(pos, h, n=None):
    """The prototype's gen_registered.register, without the files."""
    pos = np.asarray(pos)
    if n is None:
        n = int(np.ceil(max(12.0, np.ptp(pos, axis=0).max() + 7.5) / h))
        while not grid.good(n):
            n += 1
    L = n * h
    k = np.round((L / 2 - (pos.max(0) + pos.min(0)) / 2) / h)
    return n, L, pos + k * h


def stored_poscar(name):
    """The prototype's POSCAR coordinates back in molecule order, and the box."""
    lines = (PILOT / name / "POSCAR").read_text().splitlines()
    box = float(lines[2].split()[0])
    n = sum(int(v) for v in lines[6].split())
    xyz = np.array([[float(v) for v in line.split()] for line in lines[8 : 8 + n]])
    order = [int(v) for v in (PILOT / name / "order.txt").read_text().split()]
    out = np.zeros_like(xyz)
    out[order] = xyz
    return box, out


def stored_ngx(name):
    for line in (PILOT / name / "INCAR").read_text().splitlines():
        if line.startswith("NGX ="):
            return int(line.split("=")[1])


# ---- grid registration ----------------------------------------------------------
@pytest.mark.parametrize("name", ["m00", "d00_18"])
def test_registration_matches_the_prototype(name):
    pos = np.array(FRAGS["frags"][name]["pos"])
    placed = grid.register(pos, CELL)
    assert placed["reference_ng"] == [150, 150, 150]
    box, xyz = stored_poscar(f"r2_{name}")
    assert placed["ng"] == [stored_ngx(f"r2_{name}")] * 3
    assert np.allclose(placed["box"], box, atol=1e-9)
    assert np.abs(placed["coordinates"] - xyz).max() < 1e-9


def test_cell_registration_matches_the_prototype():
    placed = grid.register_cell(X, CELL)
    assert placed["ng"] == [150, 150, 150] == [stored_ngx("r2_cell")] * 3
    box, xyz = stored_poscar("r2_cell")
    assert abs(box - BOX) < 1e-9
    assert np.abs(placed["coordinates"] - xyz).max() < 1e-9


def test_outer_pair_at_an_image_registers_like_the_prototype():
    """d00_17 is an outer pair whose second molecule is at image (0, 0, -1):
    its coordinates lie outside the cell, so the shift is many grid steps."""
    frag = FRAGS["frags"]["d00_17"]
    assert frag["image"] == [0, 0, -1]
    pos = np.array(frag["pos"])
    placed = grid.register(pos, CELL)
    n, L, xyz = prototype_register(pos, BOX / 150)
    assert placed["ng"] == [n] * 3
    assert np.allclose(placed["box"], L)
    assert np.abs(placed["coordinates"] - xyz).max() < 1e-12
    shift = placed["coordinates"] - pos
    steps = shift / (BOX / 150)
    assert np.allclose(steps, np.round(steps)) and np.abs(steps).max() > 10


def test_good_grid_sizes():
    assert [n for n in range(140, 170) if grid.good(n)] == [
        140,
        144,
        150,
        160,
        162,
        168,
    ]
    assert grid.grid_points(12.4297, 0.0829) == 150
    with pytest.raises(ValueError, match="orthorhombic"):
        grid.register([[0, 0, 0]], [[10, 0, 0], [5, 9, 0], [0, 0, 10]])


# ---- get_task ------------------------------------------------------------------
@pytest.fixture
def catalog(monkeypatch, tmp_path):
    """A fake potential library with hard O and H, with ZVALs."""
    files = {}
    for name, zval, emax in (
        ("O", 6, 400),
        ("H", 1, 250),
        ("O_h", 6, 765),
        ("H_h", 1, 700),
    ):
        path = tmp_path / f"{name}.POTCAR"
        path.write_text(
            f"  PAW_PBE {name} (fake)\n   POMASS =  1.0; ZVAL   =  "
            f"{zval:.3f}    mass and valenz\n End of Dataset\n"
        )
        files[name] = {"file": str(path), "Emax": emax}
    data = {"potpaw_PBE.64": files}
    monkeypatch.setattr(batch, "potential_catalog", lambda: data)
    return data


def incar(task):
    out = {}
    for line in task.files["INCAR"].splitlines():
        key, value = line.split("=", 1)
        out[key.strip()] = value.split("#")[0].strip()
    return out


def geometry(name):
    pos = np.array(FRAGS["frags"][name]["pos"])
    return seamm_exec.Geometry([8, 1, 1] * (len(pos) // 3), pos)


def test_fragment_task(catalog):
    resources = seamm_exec.Resources(ntasks=8, mem_per_cpu=2_000_000_000)
    task = batch.get_task(
        geometry("d00_18"),
        MC,
        key="c1-d00_18",
        options={"grid": {"max_spacing": 0.0829, "reference_cell": CELL.tolist()}},
        resources=resources,
    )
    keys = incar(task)
    n = stored_ngx("r2_d00_18")
    assert keys["METAGGA"] == "R2SCAN" and keys["ENCUT"] == "1200.00"
    assert keys["NGX"] == keys["NGY"] == keys["NGZ"] == str(n)
    assert keys["NGXF"] == str(2 * n)
    assert keys["IDIPOL"] == "4" and keys["DIPOL"] == "0.5 0.5 0.5"
    assert keys["ISYM"] == "0" and keys["LWAVE"] == ".FALSE."
    assert keys["ISIF"] == "0" and keys["EDIFF"] == "1.00E-07"
    assert keys["PREC"].lower() == "accurate" and keys["ALGO"] == "All"
    assert keys["NCORE"] == "4" and keys["KPAR"] == "1"
    assert "NELECT" not in keys
    # POSCAR: Cartesian, 10 decimals, the prototype's registered coordinates
    lines = task.files["POSCAR"].splitlines()
    assert lines[7] == "Cartesian"
    xyz = np.array([[float(v) for v in line.split()] for line in lines[8:]])
    box, expected = stored_poscar("r2_d00_18")
    order = [int(v) for v in (PILOT / "r2_d00_18" / "order.txt").read_text().split()]
    assert np.abs(xyz - expected[order]).max() < 1e-9
    assert float(lines[2].split()[0]) == pytest.approx(box, abs=1e-9)
    assert task.files["POTCAR"].index("O_h") < task.files["POTCAR"].index("H_h")
    assert task.files["KPOINTS"].splitlines()[2:4] == ["Gamma", "1 1 1"]
    # D4 of the isolated fragment, after VASP, in the same task
    assert "fragment.xyz" in task.files
    assert task.cmd[0] == "{gamma_code}" and "{dftd4}" in task.cmd
    i = task.cmd.index("{dftd4}")
    assert task.cmd[i + 1] == "fragment.xyz" and "r2scan" in task.cmd
    assert task.success_text == {"OUTCAR": "General timing", "dftd4.json": "energy"}
    assert task.program == "vasp" and task.resources is resources
    assert 100 < task.estimated_seconds < 2000


def test_cell_task(catalog):
    cell = seamm_exec.Geometry([8, 1, 1] * 64, X, cell=CELL)
    task = batch.get_task(
        cell,
        MC,
        key="c1-cell",
        properties=("energy", "gradients", "stress"),
        options={"grid": {"max_spacing": 0.0829}},
        resources=seamm_exec.Resources(ntasks=16),
    )
    keys = incar(task)
    assert keys["NGX"] == "150" and keys["ISIF"] == "2"
    assert "IDIPOL" not in keys and "fragment.xyz" not in task.files
    i = task.cmd.index("{dftd4}")
    assert task.cmd[i + 1] == "POSCAR"
    lines = task.files["POSCAR"].splitlines()
    xyz = np.array([[float(v) for v in line.split()] for line in lines[8:]])
    _, expected = stored_poscar("r2_cell")
    order = [int(v) for v in (PILOT / "r2_cell" / "order.txt").read_text().split()]
    assert np.abs(xyz - expected[order]).max() < 1e-9


def test_charged_fragment_sets_nelect(catalog):
    hydroxide = seamm_exec.Geometry([8, 1], [[5, 5, 5], [5.97, 5, 5]], charge=-1)
    task = batch.get_task(
        hydroxide,
        MC,
        key="oh",
        options={"grid": {"reference_cell": CELL.tolist()}},
    )
    assert incar(task)["NELECT"] == "8.0000"  # 6 + 1 + 1 electron
    assert task.cmd[task.cmd.index("--charge") + 1] == "-1"


def test_molecule_without_reference_is_refused(catalog):
    with pytest.raises(ValueError, match="reference_cell"):
        batch.get_task(geometry("m00"), MC, key="m")
    assert not batch.can_run_task(geometry("m00"), MC)
    assert batch.can_run_task(
        geometry("m00"), MC, options={"grid": {"reference_cell": CELL.tolist()}}
    )


def test_batch_and_substep_write_the_same_inputs(catalog, tmp_path):
    """The same settings through the Energy substep and through get_task give
    the same INCAR, KPOINTS and POTCAR (the POSCAR differs only in form: the
    batch path writes Cartesian coordinates with 10 decimals)."""
    from molsystem import SystemDB
    import vasp_step

    db = SystemDB(filename="file:vasp_same?mode=memory&cache=shared")
    try:
        system, configuration = harness.water_cell(db)
        energy = vasp_step.Energy()
        for key, value in batch.SETTINGS.items():
            energy.parameters[key].value = value
        values = {
            "model": "Meta-generalized gradient approximations (meta-GGA)",
            "submodel": "r2SCAN",
            "plane-wave cutoff": 1200.0,
            "calculate stress": "yes",
            "ncore": 4,
            "potentials": {"O": "O_h", "H": "H_h"},
            "extra keywords": ["ISYM=0", "LWAVE=.FALSE.", "LCHARG=.FALSE."],
        }
        for key, value in values.items():
            energy.parameters[key].value = value
        energy._id = (1, 1)
        energy._timing_data = None
        energy.get_system_configuration = lambda *a, **k: (system, configuration)
        energy.parent = SimpleNamespace(
            potential_metadata=catalog, get_value=lambda v: v
        )
        P = energy.parameters.current_values_to_dict(context={})
        substep = energy.get_input(P)

        mc = dict(MC, method="r2SCAN", level="VASP:DFT@r2SCAN/PAW-hard@1200")
        task = batch.get_task(
            configuration,
            mc,
            key="same",
            properties=("energy", "gradients", "stress"),
            resources=seamm_exec.Resources(ntasks=8),
        )
    finally:
        db.close()
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        assert task.files[name] == substep[name], name


# ---- analyze_task ----------------------------------------------------------------
def result_for(name):
    files = {}
    for f in ("vasprun.xml", "OUTCAR"):
        files[f] = gzip.open(PILOT / name / f"{f}.gz", "rt").read()
    files["dftd4.json"] = (PILOT / name / "dftd4.json").read_text()
    return SimpleNamespace(key=name, files=files)


@pytest.mark.parametrize("name", ["m00", "d00_18"])
def test_analyze_fragment(name):
    expected = EXPECTED[name]
    out = batch.analyze_task(result_for(f"r2_{name}"), MC, geometry(name))
    assert out["energy"] == pytest.approx(
        (expected["vasp_E"] + expected["d4_E"]) * EV_KJ, abs=1e-6
    )
    # vasprun.xml has 8 decimals in the forces, OUTCAR (the prototype's) 6
    forces = np.array(expected["vasp_F"]) + np.array(expected["d4_F"])
    assert np.abs(out["gradients"] + forces * EV_KJ).max() < 1e-4
    assert "stress" not in out


def test_analyze_cell_and_the_sign_of_the_stress():
    """The stress is sigma = -P in GPa, VASP's "in kB" pressure and dftd4's
    virial (dE/dstrain) combined: the pilot cell's P_VASP + P_D4."""
    expected = EXPECTED["cell"]
    cell = seamm_exec.Geometry([8, 1, 1] * 64, X, cell=CELL)
    out = batch.analyze_task(
        result_for("r2_cell"), MC, cell, properties=("energy", "gradients", "stress")
    )
    assert out["energy"] == pytest.approx(
        (expected["vasp_E"] + expected["d4_E"]) * EV_KJ, abs=1e-5
    )
    forces = np.array(expected["vasp_F"]) + np.array(expected["d4_F"])
    assert np.abs(out["gradients"] + forces * EV_KJ).max() < 1e-4  # 6 vs 8 decimals

    xx, yy, zz, xy, yz, zx = expected["stress_kB"]
    p_vasp = 0.1 * np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])  # GPa
    volume = BOX**3
    w_d4 = -np.array(expected["d4_dEdstrain"])  # eV
    p_d4 = w_d4 / volume * Q_(1.0, "eV/Å^3").m_as("GPa")
    sigma = np.array(out["stress"])
    # vasprun.xml's tensor is not exactly symmetric (2e-6 GPa here); the
    # prototype's "in kB" line is
    assert np.allclose(sigma, -(p_vasp + p_d4), atol=5e-6)
    atm = -np.trace(sigma) / 3 * 1e9 / 101325
    # The prototype's P_VASP + P_D4 (exact kB -> atm): -61519.55 - 2322.50
    assert atm == pytest.approx(-61519.546 - 2322.500, abs=0.01)


def test_final_energy_not_an_scf_step():
    energy, forces, stress = batch.parse_vasprun(
        result_for("r2_cell").files["vasprun.xml"]
    )
    assert energy == pytest.approx(EXPECTED["cell"]["vasp_E"], abs=1e-7)
    assert forces.shape == (192, 3) and stress.shape == (3, 3)


def test_unconverged_scf_is_refused():
    result = result_for("r2_m00")
    result.files["OUTCAR"] = result.files["OUTCAR"].replace(
        "aborting loop because EDIFF is reached", "aborting loop"
    )
    with pytest.raises(AnalysisError, match="did not converge"):
        batch.analyze_task(result, MC, geometry("m00"))


# ---- model chemistries and the resolver ----------------------------------------
def test_model_chemistry_options():
    options = batch.get_model_chemistry_options()
    entry = options["r2SCAN-D4"]
    assert entry["model_chemistry"] == "VASP:DFT@r2SCAN-D4/PAW"
    assert entry["stress_convention"] == "stress" and entry["prefers_batch"]
    assert entry["periodic_native"] and not entry["mdi_capable"]
    assert batch.get_model_chemistry_options(mdi_only=True) == {}
    from model_chemistry_step.grammar import parse_level

    parsed = parse_level("VASP:DFT@r2SCAN-D4/PAW-hard@1200")
    assert (parsed["method"], parsed["basis"], parsed["cutoff"]) == (
        "r2SCAN-D4",
        "PAW-hard",
        "1200",
    )


def test_resolver():
    config, cmd, env = resolver.resolve(
        {"gamma_code": "mpiexec -np {NTASKS} vasp_gam", "dftd4": "/opt/dftd4"},
        ["{gamma_code}", ">", "vasp.out", "&&", "{dftd4}", "POSCAR"],
        {},
        {"NTASKS": 8},
        "/root",
    )
    assert cmd == [
        "mpiexec -np {NTASKS} vasp_gam",
        ">",
        "vasp.out",
        "&&",
        "/opt/dftd4",
        "POSCAR",
    ]
    with pytest.raises(RuntimeError, match="dftd4"):
        resolver.resolve({"gamma_code": "x", "dftd4": ""}, ["{dftd4}"], {}, {}, "/r")


def test_d4_never_doubles_vasps_own_dispersion(catalog):
    """revPBE's metadata carries IVDW = 12; revPBE-D4 must not keep it."""
    import vasp_step

    dft = vasp_step.metadata["computational models"]["Density Functional Theory (DFT)"]
    revpbe = dft["models"]["Generalized-gradient approximations (GGA)"][
        "parameterizations"
    ]["revPBE : the revised PBE functional of Zhang and Yang"]
    assert revpbe["keywords"].get("IVDW") == "12"  # the metadata as it is
    mc = dict(MC, method="revPBE-D4", basis="PAW", cutoff="500")
    task = batch.get_task(
        geometry("m00"),
        mc,
        key="m",
        options={"grid": {"reference_cell": CELL.tolist()}},
    )
    keys = incar(task)
    assert keys["GGA"] == "RE"
    assert "IVDW" not in keys and not any(k.startswith("VDW_") for k in keys)
    assert "{dftd4}" in task.cmd and "revpbe" in task.cmd
