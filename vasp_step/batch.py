# -*- coding: utf-8 -*-

"""VASP's side of the Model Chemistry batch contract.

``get_task`` writes the same inputs the Energy substep writes for the same
settings (both use :mod:`vasp_step.inputs`), and ``analyze_task`` reads the
energy, forces and stress back. The model chemistries are

    VASP:DFT@<functional>/<potentials>@<ENCUT in eV>

for example ``VASP:DFT@r2SCAN-D4/PAW-hard@1200``. The potentials are a PAW set,
optionally a named variant (``PAW`` = potpaw_PBE.64 with VASP's recommended
potentials, ``PAW-hard`` = the same with the hard potentials where they exist,
``PAW-LDA`` = potpaw_LDA.64). Without a cutoff, ENCUT is 1.3 × the largest
ENMAX of the potentials.

Dispersion. A functional with ``-D4`` is the functional plus the dftd4 program's
D4 correction for it, computed in the same task after VASP (``dftd4`` in
vasp.ini), because VASP builds are often compiled without D4 (IVDW = 13). A
periodic cell gets the periodic D4; a fragment in a box gets the D4 of the
isolated fragment, as its molecular counterparts do, without the dispersion
with its images.

``options`` for a structure:

* ``grid``: ``{"max_spacing": Å}`` sets the cell's FFT grid explicitly (NGX,
  NGXF = 2·NGX, ...), the cell's atoms shifted by whole grid steps. With
  ``"reference_cell"`` (3x3) a molecular structure is a fragment registered on
  that parent cell's grid (see :mod:`vasp_step.grid`), in a box of its extent
  plus ``"padding"`` (7.5 Å), with the dipole correction. Without ``grid`` a
  molecular structure cannot run.
* ``charge``, ``multiplicity``: override the structure's.
* ``dipole``: False to leave out the dipole correction of a fragment.

The stress comes back in GPa as a stress (sigma = -P, tensile positive), as the
Energy substep stores it, so the model chemistries declare
``stress_convention = "stress"``; with D4 it includes D4's.

The POTCARs are read where the task is built, from the VASP potential library
(``<SEAMM>/Parameters/VASP``, catalogued by the VASP step) on that machine.
"""

import json
import math

import numpy as np

import seamm_exec
from seamm_exec.evaluator import AnalysisError, check_properties, structure_data
from seamm_util import Q_

from . import grid as grid_
from . import inputs

#: The PAW sets the model chemistries name: name -> (set, variant)
POTENTIAL_SETS = {
    "PAW": ("potpaw_PBE.64", None),
    "PAW-hard": ("potpaw_PBE.64", "hard"),
    "PAW-LDA": ("potpaw_LDA.64", None),
}

#: dftd4's name for the functionals that have D4 parameters
DFTD4_FUNCTIONALS = {
    "PBE": "pbe",
    "RPBE": "rpbe",
    "revPBE": "revpbe",
    "TPSS": "tpss",
    "SCAN": "scan",
    "r2SCAN": "r2scan",
    "B3LYP": "b3lyp",
    "PBE0": "pbe0",
    "HSE06": "hse06",
}

#: Energy-parameter settings of a batch calculation (on top of the defaults)
SETTINGS = {
    "k-grid method": "𝚪-point",
    "precision": "accurate",
    "electronic method": "all",
    "ediff": 1.0e-7,
    "nelm": 200,
    "smearing width": 0.01,
    "lreal": "no",
    "lorbit": "no",
    "kpar": 1,
    "use hdf5 files": "no",
}

HARTREE_EV = Q_(1.0, "E_h").m_as("eV")
EV_KJ = Q_(1.0, "eV").m_as("kJ/mol")
BOHR = Q_(1.0, "bohr").m_as("Å")


def functionals():
    """The functionals: name -> (model, submodel, dftd4 name or None)."""
    import vasp_step

    dft = vasp_step.metadata["computational models"]["Density Functional Theory (DFT)"]
    result = {}
    for model, data in dft["models"].items():
        for submodel in data["parameterizations"]:
            name = submodel.split(" : ")[0].strip()
            result[name] = (model, submodel, None)
            if name in DFTD4_FUNCTIONALS:
                result[f"{name}-D4"] = (model, submodel, DFTD4_FUNCTIONALS[name])
    return result


def get_model_chemistry_options(periodic_only=False, mdi_only=False):
    """VASP's model chemistries (all periodic, none through MDI)."""
    if mdi_only:
        return {}
    options = {}
    for name in functionals():
        potentials = (
            "PAW-LDA" if "LDA" in name or name in ("VWN5", "PW92", "PZ-LDA") else "PAW"
        )
        options[name] = {
            "model_chemistry": f"VASP:DFT@{name}/{potentials}",
            "type": "DFT",
            "description": "",
            "periodic_native": True,
            "periodic_mdi": False,
            "elements": "1-94",
            "mdi_capable": False,
            "mdi_method_arg": None,
            "prefers_batch": True,
            "stress_convention": "stress",
        }
    return options


def potential_catalog():
    """The catalogue of the VASP potential library on this machine."""
    from seamm_util import installation_path

    index = installation_path("Parameters", "VASP") / "index.json"
    if not index.exists():
        raise RuntimeError(
            f"The VASP potential library has no catalogue ({index}). Run a VASP "
            "step once on this machine to create it."
        )
    return json.loads(index.read_text())


def _level(model_chemistry):
    """(functional name, set, variant, ENCUT or None) of a model chemistry."""
    method = model_chemistry.get("method")
    table = functionals()
    if method not in table:
        raise ValueError(
            f"VASP has no functional '{method}'. The functionals are: "
            + ", ".join(sorted(table))
        )
    basis = model_chemistry.get("basis") or "PAW"
    if basis not in POTENTIAL_SETS:
        raise ValueError(
            f"Unknown VASP potentials '{basis}': one of {sorted(POTENTIAL_SETS)}."
        )
    potential_set, variant = POTENTIAL_SETS[basis]
    cutoff = model_chemistry.get("cutoff")
    encut = None
    if cutoff not in (None, ""):
        encut = float(str(cutoff).lower().replace("ev", ""))
    return method, potential_set, variant, encut


def can_run_task(configuration, model_chemistry, *, options=None):
    """A periodic structure, or a molecule registered on a parent cell."""
    options = options or {}
    data = structure_data(configuration)
    if data["periodicity"] == 3:
        return True
    return data["periodicity"] == 0 and "reference_cell" in (options.get("grid") or {})


def estimated_seconds(n_atoms, ng, ntasks):
    """A cost estimate calibrated on the MBE prototype: a water fragment in a
    150³ grid took ~330 s on 8 ranks of TinkerCliffs, the 192-atom cell ~1100 s
    on 16."""
    points = float(np.prod(ng)) / 150**3 if ng else 1.0
    return 330.0 * points * math.sqrt(max(n_atoms, 3) / 6.0) * 8.0 / max(1, ntasks)


def _zval(potcar):
    """The valence charge of each potential in a POTCAR, in order."""
    values = []
    for line in potcar.splitlines():
        if "ZVAL" in line:
            values.append(float(line.split("ZVAL")[1].split("=")[1].split()[0]))
    return values


def get_task(
    configuration,
    model_chemistry,
    *,
    key,
    properties=("energy", "gradients"),
    options=None,
    resources=None,
):
    """A :class:`seamm_exec.Task` computing VASP's energy, forces and (for a
    cell, if asked) stress. See the module docstring."""
    import vasp_step

    options = dict(options or {})
    data = structure_data(configuration)
    method, potential_set, variant, encut = _level(model_chemistry)
    model, submodel, d4 = functionals()[method]
    grid_options = dict(options.get("grid") or {})
    periodic = data["periodicity"] == 3
    if not periodic and "reference_cell" not in grid_options:
        raise ValueError(
            "VASP calculations are periodic: a molecule needs grid['reference_cell'] "
            "to be placed in a box registered on its parent cell."
        )
    charge = int(options.get("charge", data["charge"]))
    multiplicity = int(options.get("multiplicity", data["multiplicity"]))
    atnos = list(data["atomic_numbers"])
    xyz = np.asarray(data["coordinates"], dtype=float)
    max_spacing = float(grid_options.get("max_spacing", grid_.DEFAULT_MAX_SPACING))

    ng = None
    dipole = False
    if periodic:
        cell = np.asarray(data["cell"], dtype=float)
        if grid_options:
            placed = grid_.register_cell(xyz, cell, max_spacing)
            xyz, ng = placed["coordinates"], placed["ng"]
    else:
        placed = grid_.register(
            xyz,
            grid_options["reference_cell"],
            max_spacing,
            float(grid_options.get("padding", grid_.DEFAULT_PADDING)),
        )
        xyz, ng = placed["coordinates"], placed["ng"]
        cell = np.diag(placed["box"])
        dipole = options.get("dipole", True)

    catalog = potential_catalog()[potential_set]
    potcar, names = inputs.potcar_text(atnos, potential_set, catalog, variant=variant)
    if encut is None:
        encut = 1.3 * inputs.enmax(atnos, potential_set, catalog, variant=variant)

    ntasks = 1 if resources is None or not resources.ntasks else int(resources.ntasks)
    # The same conversion of the values as the substep's (e.g. "no" -> False)
    parameters = vasp_step.EnergyParameters()
    values = dict(SETTINGS)
    values["model"], values["submodel"] = model, submodel
    values["calculate stress"] = (
        "yes" if (periodic and "stress" in properties) else "no"
    )
    values["ncore"] = 4 if ntasks % 4 == 0 else 1
    if multiplicity != 1:
        values["spin polarization"] = "collinear"
    for name, value in values.items():
        parameters[name].value = value
    P = parameters.current_values_to_dict(context={})

    extra = [("ISYM", 0), ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE.")]
    if charge != 0:
        counts = inputs.atom_order(atnos)[2]
        unique = sorted(set(atnos), reverse=True)
        nelect = sum(z * counts[a] for z, a in zip(_zval(potcar), unique)) - charge
        extra.append(("NELECT", f"{nelect:.4f}"))
    if multiplicity != 1:
        extra.append(("NUPDOWN", multiplicity - 1))
    if ng is not None:
        extra += [("NGX", ng[0]), ("NGY", ng[1]), ("NGZ", ng[2])]
        extra += [("NGXF", 2 * ng[0]), ("NGYF", 2 * ng[1]), ("NGZF", 2 * ng[2])]
    if dipole:
        extra += [("IDIPOL", 4), ("LDIPOL", ".TRUE."), ("DIPOL", "0.5 0.5 0.5")]

    functional = vasp_step.metadata["computational models"][
        "Density Functional Theory (DFT)"
    ]["models"][model]["parameterizations"][submodel]
    if d4:
        # dftd4 adds the dispersion: VASP must not add its own as well (plain
        # revPBE's metadata carries IVDW = 12, for one)
        functional = dict(functional)
        functional["keywords"] = {
            k: v
            for k, v in functional["keywords"].items()
            if k != "IVDW" and not k.startswith("VDW_")
        }
    keywords, descriptions = inputs.keywords(
        P,
        functional=functional,
        istart=0,
        encut=encut,
        extra=extra,
        keyword_metadata=vasp_step.metadata["keywords"],
    )
    kpoints, _ = inputs.kpoints_text(P)
    files = {
        "INCAR": inputs.incar_text(
            keywords, descriptions, vasp_step.metadata["keywords"]
        ),
        "POTCAR": potcar,
        "KPOINTS": kpoints,
        "POSCAR": inputs.poscar_text(key, cell, atnos, xyz, cartesian=True, digits=10),
    }

    cmd = ["{gamma_code}", ">", "vasp.out", "2>&1"]
    success = {"OUTCAR": "General timing"}
    return_files = [
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "OUTCAR",
        "OSZICAR",
        "vasprun.xml",
        "vasp.out",
    ]
    if d4:
        d4_input = "POSCAR"
        if not periodic:
            symbols = data["symbols"]
            files["fragment.xyz"] = f"{len(atnos)}\n{key}\n" + "".join(
                f"{s} {x:.10f} {y:.10f} {z:.10f}\n"
                for s, (x, y, z) in zip(symbols, xyz)
            )
            d4_input = "fragment.xyz"
        cmd += [
            "&&",
            "{dftd4}",
            d4_input,
            "--func",
            d4,
            "--grad",
            "--json",
            "dftd4.json",
            "--noedisp",
            "--charge",
            str(charge),
            ">",
            "dftd4.out",
        ]
        success["dftd4.json"] = "energy"
        return_files += ["dftd4.json", "dftd4.out", "fragment.xyz"]

    if resources is None:
        resources = seamm_exec.Resources(ntasks=ntasks, mem_per_cpu=2_000_000_000)
    return seamm_exec.Task(
        key=key,
        program="vasp",
        cmd=cmd,
        shell=True,
        files=files,
        return_files=return_files,
        resources=resources,
        estimated_seconds=estimated_seconds(len(atnos), ng, ntasks),
        success_text=success,
    )


def parse_vasprun(text):
    """The final energy (sigma -> 0, eV), forces (eV/Å, VASP order) and stress
    (kB as VASP prints it, a pressure, or None) from vasprun.xml."""
    from lxml import etree

    root = etree.fromstring(text.encode() if isinstance(text, str) else text)
    calculation = root.findall("calculation")[-1]
    # The calculation's own energy block, not one of its SCF steps'
    energy = calculation.find("energy")
    e0 = float(energy.find("i[@name='e_0_energy']").text)

    def array(name):
        node = calculation.find(f"varray[@name='{name}']")
        if node is None:
            return None
        return np.array([[float(x) for x in v.text.split()] for v in node.findall("v")])

    return e0, array("forces"), array("stress")


def converged(outcar):
    """Whether VASP's SCF reached EDIFF (an unconverged run must not count)."""
    return "aborting loop because EDIFF is reached" in outcar


def analyze_task(
    result,
    model_chemistry,
    configuration,
    *,
    properties=("energy", "gradients"),
    options=None,
):
    """{"energy": kJ/mol, "gradients": (n, 3) kJ/mol/Å, "stress": (3, 3) GPa,
    sigma = -P} of a finished task, in the structure's atom order."""
    data = structure_data(configuration)
    atnos = list(data["atomic_numbers"])
    periodic = data["periodicity"] == 3
    outcar = _text(result, "OUTCAR")
    if not converged(outcar):
        raise AnalysisError(f"'{result.key}': the SCF did not converge (EDIFF)")
    energy, forces, stress = parse_vasprun(_text(result, "vasprun.xml"))
    _, to_seamm, _ = inputs.atom_order(atnos)
    ordered = np.zeros_like(forces)
    ordered[to_seamm] = forces
    out = {"energy": energy * EV_KJ, "gradients": -ordered * EV_KJ}
    sigma = None
    if stress is not None and periodic:
        sigma = -0.1 * stress  # kB pressure -> GPa stress
    method, *_ = _level(model_chemistry)
    if functionals()[method][2]:
        d4 = json.loads(_text(result, "dftd4.json"))
        out["energy"] += d4["energy"] * HARTREE_EV * EV_KJ
        gradient = np.array(d4["gradient"]).reshape(-1, 3) * HARTREE_EV / BOHR
        if periodic:  # dftd4 read the POSCAR: VASP order
            g = np.zeros_like(gradient)
            g[to_seamm] = gradient
            gradient = g
        out["gradients"] = out["gradients"] + gradient * EV_KJ
        if sigma is not None:
            volume = abs(np.linalg.det(np.asarray(data["cell"], dtype=float)))
            virial = np.array(d4["virial"]).reshape(3, 3) * HARTREE_EV  # dE/dstrain
            sigma = sigma + virial / volume * Q_(1.0, "eV/Å^3").m_as("GPa")
    if sigma is not None and "stress" in properties:
        out["stress"] = sigma.tolist()
    check_properties(out, properties, f"'{result.key}'", periodic=periodic)
    return out


def _text(result, name):
    value = result.files.get(name)
    if value is None:
        raise AnalysisError(f"'{result.key}': no {name} came back")
    return value.decode() if isinstance(value, bytes) else value
