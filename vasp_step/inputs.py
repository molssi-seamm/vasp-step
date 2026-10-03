# -*- coding: utf-8 -*-

"""The VASP input files from explicit settings, with no flowchart.

The Energy substep and the batch path (``get_task``) both build their INCAR,
POTCAR, KPOINTS and POSCAR here, so the same settings give the same inputs.
``P`` is a dictionary of the Energy parameters' values (as
``EnergyParameters().current_values_to_dict()`` gives them); anything that
needs the flowchart -- the initial wavefunction, an ENCUT given as an
expression, extra keywords with variables -- is resolved by the caller and
passed in.
"""

from math import ceil as ceiling
from pathlib import Path

import molsystem

from .potentials import potentials_for


def atom_order(atomic_numbers):
    """VASP's atom order: by descending atomic number, stable within an element.

    Returns
    -------
    to_vasp : [int]
        The VASP position of each atom.
    to_seamm : [int]
        The original index of each VASP position.
    element_count : {int: int}
        The number of atoms of each atomic number.
    """
    atnos = list(atomic_numbers)
    unique = sorted(set(atnos), reverse=True)
    count = {atno: 0 for atno in atnos}
    for atno in atnos:
        count[atno] += 1
    offset = {}
    n = 0
    for atno in unique:
        offset[atno] = n
        n += count[atno]
    to_vasp = []
    to_seamm = [-1] * len(atnos)
    for original, atno in enumerate(atnos):
        new = offset[atno]
        to_vasp.append(new)
        to_seamm[new] = original
        offset[atno] += 1
    return to_vasp, to_seamm, count


def encut_value(value, enmax, context=None):
    """ENCUT in eV from the parameter: a quantity, a number, or an expression
    of ENMAX (e.g. "1.3*ENMAX") evaluated with ``context`` (the flowchart's
    variables)."""
    if isinstance(value, str):
        scope = {**(context or {}), "ENMAX": enmax, "enmax": enmax}
        return eval(value, scope)  # noqa: S307 -- a flowchart expression
    if hasattr(value, "m_as"):
        return value.m_as("eV")
    return float(value)


def keywords(P, *, functional, istart, encut, extra=(), keyword_metadata=None):
    """The INCAR keywords and their descriptions.

    Parameters
    ----------
    P : dict
        The Energy parameters' values.
    functional : dict
        The functional's metadata entry: "keywords" and "description".
    istart : int
        0 (from scratch) or 1 (read WAVECAR).
    encut : float
        The plane-wave cutoff, eV.
    extra : [(str, object)]
        Extra keywords, already dereferenced, replacing or adding to the rest.
    keyword_metadata : dict
        The INCAR keyword descriptions (metadata["keywords"]).

    Returns
    -------
    keywords : dict
    descriptions : dict
    """
    keyword_metadata = keyword_metadata or {}
    descriptions = {}
    result = {}

    tmp = functional["keywords"]
    result.update(tmp)
    descriptions[list(tmp)[0]] = functional["description"]

    if P["spin polarization"] == "collinear":
        result["ISPIN"] = 2
    elif P["spin polarization"] == "noncollinear":
        result["LNONCOLLINEAR"] = ".True."
    else:
        result["ISPIN"] = 1

    result["LASPH"] = ".True." if P["nonspherical PAW"] else ".False."
    result["ENCUT"] = f"{encut:.2f}"
    result["ISTART"] = istart

    result["ALGO"] = P["electronic method"].title().replace(" ", "")
    result["ISEARCH"] = 1
    result["NELM"] = P["nelm"]
    result["NELMIN"] = 2 if P["nelmin"] == "default" else P["nelmin"]
    result["EDIFF"] = f'{P["ediff"]:.2E}'
    result["PREC"] = P["precision"]

    _type = P["occupation type"].lower()
    if "gaussian" in _type:
        ismear = 0
    elif "methfessel" in _type:
        ismear = P["Methfessel-Paxton order"]
    elif "tetrahedron" in _type:
        if "corrections" in _type:
            ismear = -15 if "fermi" in _type else -5
        else:
            ismear = -14 if "fermi" in _type else -4
    elif "fermi" in _type:
        ismear = -1
    else:
        raise ValueError(f"Occupation type (ISMEAR) '{_type} not recognized.")
    result["ISMEAR"] = ismear
    descriptions["ISMEAR"] = _type
    if ismear >= -1 or ismear in (-15, -14):
        sigma = P["smearing width"].m_as("eV")
        result["SIGMA"] = f"{sigma:.2f}"

    result["IBRION"] = -1
    match P["calculate stress"]:
        case "no":
            isif = 0
        case "only pressure":
            isif = 1
        case _:
            isif = 2
    result["ISIF"] = isif
    result["NSW"] = 0
    efermi = P["efermi"]
    if "middle" in efermi:
        result["EFERMI"] = "MIDGAP"
    elif efermi == "legacy":
        result["EFERMI"] = "Legacy"
    else:
        result["EFERMI"] = efermi.m_as("eV")

    result["LH5"] = ".True." if P["use hdf5 files"] else ".False."
    if P["lorbit"]:
        result["LORBIT"] = 11

    result["NCORE"] = P["ncore"]
    result["KPAR"] = P["kpar"]
    result["LPLANE"] = ".True." if P["lplane"] else ".False."
    result["LREAL"] = "Auto" if P["lreal"] else ".False."
    result["NSIM"] = P["nsim"]
    result["LSCALAPACK"] = ".True." if P["lscalapack"] else ".False."
    if P["lscalapack"]:
        result["LSCALU"] = ".True." if P["lscalu"] else ".False."

    for key, value in extra:
        result[key] = value
        if key in keyword_metadata:
            descriptions[key] = keyword_metadata[key]["description"]

    return result, descriptions


def incar_text(keywords, descriptions, keyword_metadata=None):
    """The INCAR file, one keyword per line with its description."""
    keyword_metadata = keyword_metadata or {}
    lines = []
    for key, value in keywords.items():
        if key in descriptions:
            lines.append(f"{key:>20s} = {value:<20}  # {descriptions[key]}")
        elif key in keyword_metadata and "description" in keyword_metadata[key]:
            lines.append(
                f"{key:>20s} = {value:<20}  # {keyword_metadata[key]['description']}"
            )
        else:
            lines.append(f"{key:>20s} = {value}")
    return "\n".join(lines)


def potcar_text(
    atomic_numbers, potential_set, potential_data, chosen=None, variant=None
):
    """The POTCAR: the potentials of the elements by descending atomic number.

    Parameters
    ----------
    atomic_numbers : [int]
    potential_set : str
        e.g. "potpaw_PBE.64".
    potential_data : dict
        The catalog of the set: name -> {"file": path, "Emax": ...}.
    chosen : {str: str}
        Potentials chosen by element; the others are the set's defaults (or
        ``variant``'s).
    variant : str or None
        e.g. "hard".

    Returns
    -------
    text : str
    names : [str]
        The potentials, in POTCAR order.
    """
    atnos = sorted(set(atomic_numbers), reverse=True)
    elements = molsystem.elements.to_symbols(atnos)
    names = potentials_for(potential_set, elements, chosen, variant)
    text = ""
    for element in elements:
        text += Path(potential_data[names[element]]["file"]).read_text()
    return text, [names[e] for e in elements]


def enmax(atomic_numbers, potential_set, potential_data, chosen=None, variant=None):
    """The largest ENMAX (eV) of the potentials used."""
    atnos = sorted(set(atomic_numbers), reverse=True)
    elements = molsystem.elements.to_symbols(atnos)
    names = potentials_for(potential_set, elements, chosen, variant)
    return max(float(potential_data[names[e]]["Emax"]) for e in elements)


def kpoints_text(P, reciprocal_lengths=None):
    """The KPOINTS file and whether it is the Gamma point only."""
    lines = []
    if "point" in P["k-grid method"]:
        lines.append("𝚪-point only")
        na = nb = nc = 1
    elif "explicit" in P["k-grid method"]:
        lines.append("Explicit k-point mesh")
        na = P["na"]
        nb = P["nb"]
        nc = P["nc"]
    else:
        spacing = P["k-spacing"].to("1/Å").magnitude
        lines.append(f"k-point mesh with spacing {spacing}")
        na = max(1, ceiling(reciprocal_lengths[0] / spacing))
        nb = max(1, ceiling(reciprocal_lengths[1] / spacing))
        nc = max(1, ceiling(reciprocal_lengths[2] / spacing))
        if P["odd grid"]:
            na = na + 1 if na % 2 == 0 else na
            nb = nb + 1 if nb % 2 == 0 else nb
            nc = nc + 1 if nc % 2 == 0 else nc
    gamma_only = na == 1 and nb == 1 and nc == 1

    lines.append("0")
    if "Monkhorst" in P["centering"] and "point" not in P["k-grid method"]:
        lines.append("Monkhorst-Pack")
    else:
        lines.append("Gamma")
    lines.append(f"{na} {nb} {nc}")
    lines.append("0 0 0")
    return "\n".join(lines), gamma_only


def poscar_title(system_name, configuration_name, formula):
    """The POSCAR title line, as SEAMM writes it.

    ``formula`` is ``configuration.formula``: (formula, empirical, Z).
    """
    if system_name == "" and configuration_name == "":
        text, empirical, Z = formula
        return text if Z == 1 else f"({empirical}) * {Z}"
    title = system_name + "/" + configuration_name
    if len(title) > 100:
        if len(configuration_name) <= 100:
            title = configuration_name
        else:
            text, empirical, Z = formula
            title = text if Z == 1 else f"({empirical}) * {Z}"
    return title


def poscar_text(
    title, vectors, atomic_numbers, coordinates, *, cartesian=False, digits=6
):
    """The POSCAR file.

    Parameters
    ----------
    title : str
    vectors : [[float]]
        The lattice vectors (Å), one per row.
    atomic_numbers : [int]
        In the caller's atom order.
    coordinates : [[float]]
        Fractional coordinates, or Cartesian (Å) with ``cartesian``, in the
        caller's atom order; written in VASP's order.
    cartesian : bool
        Write Cartesian coordinates.
    digits : int
        Decimal places of the cell and the coordinates (6 as the substep
        writes; a registered fragment needs ~10 so that its box is an exact
        multiple of the reference grid step).
    """
    width = digits + 6
    lines = [title, "1.0"]
    for a, b, c in vectors:
        lines.append(
            f"{a:{width}.{digits}f} {b:{width}.{digits}f} {c:{width}.{digits}f}"
        )

    _, to_seamm, count = atom_order(atomic_numbers)
    unique = sorted(set(atomic_numbers), reverse=True)
    elements = molsystem.elements.to_symbols(unique)
    lines.append(" ".join(f"{el:>3s}" for el in elements))
    lines.append(" ".join(f"{count[atno]:3d}" for atno in unique))

    lines.append("Cartesian" if cartesian else "Direct")
    for i_seamm in to_seamm:
        lines.append(" ".join(f"{x:{width}.{digits}f}" for x in coordinates[i_seamm]))
    return "\n".join(lines)
