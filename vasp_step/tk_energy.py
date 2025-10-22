# -*- coding: utf-8 -*-

"""The graphical part of a Energy step"""

import json
from pathlib import Path
import pprint  # noqa: F401
import tkinter as tk
import tkinter.ttk as ttk

import vasp_step  # noqa: F401, E999
import seamm
from seamm_util import ureg, Q_, units_class, CompactJSONEncoder  # noqa: F401, E999
import seamm_widgets as sw

default_potentials = {
    "potpaw_PBE.64": {
        "H": "H",
        "He": "He",
        "Li": "Li_sv",
        "Be": "Be",
        "B": "B",
        "C": "C",
        "N": "N",
        "O": "O",
        "F": "F",
        "Ne": "Ne",
        "Na": "Na_pv",
        "Mg": "Mg",
        "Al": "Al",
        "Si": "Si",
        "P": "P",
        "S": "S",
        "Cl": "Cl",
        "Ar": "Ar",
        "K": "K_sv",
        "Ca": "Ca_sv",
        "Sc": "Sc_sv",
        "Ti": "Ti_sv",
        "V": "V_sv",
        "Cr": "Cr_pv",
        "Mn": "Mn_pv",
        "Fe": "Fe",
        "Co": "Co",
        "Ni": "Ni",
        "Cu": "Cu",
        "Zn": "Zn",
        "Ga": "Ga_d",
        "Ge": "Ge_d",
        "As": "As",
        "Se": "Se",
        "Br": "Br",
        "Kr": "Kr",
        "Rb": "Rb_sv",
        "Sr": "Sr_sv",
        "Y": "Y_sv",
        "Zr": "Zr_sv",
        "Nb": "Nb_sv",
        "Mo": "Mo_sv",
        "Tc": "Tc_pv",
        "Ru": "Ru_pv",
        "Rh": "Rh_pv",
        "Pd": "Pd",
        "Ag": "Ag",
        "Cd": "Cd",
        "In": "In_d",
        "Sn": "Sn_d",
        "Sb": "Sb",
        "Te": "Te",
        "I": "I",
        "Xe": "Xe",
        "Cs": "Cs_sv",
        "Ba": "Ba_sv",
        "La": "La",
        "Ce": "Ce",
        "Pr": "Pr_3",
        "Nd": "Nd_3",
        "Pm": "Pm_3",
        "Sm": "Sm_3",
        "Eu": "Eu_2",
        "Gd": "Gd_3",
        "Tb": "Tb_3",
        "Dy": "Dy_3",
        "Ho": "Ho_3",
        "Er": "Er_3",
        "Tm": "Tm_3",
        "Yb": "Yb_2",
        "Lu": "Lu_3",
        "Hf": "Hf_pv",
        "Ta": "Ta_pv",
        "W": "W_sv",
        "Re": "Re",
        "Os": "Os",
        "Ir": "Ir",
        "Pt": "Pt",
        "Au": "Au",
        "Hg": "Hg",
        "Tl": "Tl_d",
        "Pb": "Pb_d",
        "Bi": "Bi_d",
        "Po": "Po_d",
        "At": "At",
        "Rn": "Rn",
        "Fr": "Fr_sv",
        "Ra": "Ra_sv",
        "Ac": "Ac",
        "Th": "Th",
        "Pa": "Pa",
        "U": "U",
        "Np": "Np",
        "Pu": "Pu",
        "Am": "Am",
        "Cm": "Cm",
        "Cf": "Cf",
    },
    "potpaw_LDA.64": {
        "H": "H",
        "He": "He",
        "Li": "Li_sv",
        "Be": "Be",
        "B": "B",
        "C": "C",
        "N": "N",
        "O": "O",
        "F": "F",
        "Ne": "Ne",
        "Na": "Na_pv",
        "Mg": "Mg",
        "Al": "Al",
        "Si": "Si",
        "P": "P",
        "S": "S",
        "Cl": "Cl",
        "Ar": "Ar",
        "K": "K_sv",
        "Ca": "Ca_sv",
        "Sc": "Sc_sv",
        "Ti": "Ti_sv",
        "V": "V_sv",
        "Cr": "Cr_pv",
        "Mn": "Mn_pv",
        "Fe": "Fe",
        "Co": "Co",
        "Ni": "Ni",
        "Cu": "Cu",
        "Zn": "Zn",
        "Ga": "Ga_d",
        "Ge": "Ge_d",
        "As": "As",
        "Se": "Se",
        "Br": "Br",
        "Kr": "Kr",
        "Rb": "Rb_sv",
        "Sr": "Sr_sv",
        "Y": "Y_sv",
        "Zr": "Zr_sv",
        "Nb": "Nb_sv",
        "Mo": "Mo_sv",
        "Tc": "Tc_pv",
        "Ru": "Ru_pv",
        "Rh": "Rh_pv",
        "Pd": "Pd",
        "Ag": "Ag",
        "Cd": "Cd",
        "In": "In_d",
        "Sn": "Sn_d",
        "Sb": "Sb",
        "Te": "Te",
        "I": "I",
        "Xe": "Xe",
        "Cs": "Cs_sv",
        "Ba": "Ba_sv",
        "La": "La",
        "Ce": "Ce",
        "Pr": "Pr_3",
        "Nd": "Nd_3",
        "Pm": "Pm_3",
        "Sm": "Sm_3",
        "Eu": "Eu_2",
        "Gd": "Gd_3",
        "Tb": "Tb_3",
        "Dy": "Dy_3",
        "Ho": "Ho_3",
        "Er": "Er_3",
        "Tm": "Tm_3",
        "Yb": "Yb_2",
        "Lu": "Lu_3",
        "Hf": "Hf_pv",
        "Ta": "Ta_pv",
        "W": "W_sv",
        "Re": "Re",
        "Os": "Os",
        "Ir": "Ir",
        "Pt": "Pt",
        "Au": "Au",
        "Hg": "Hg",
        "Tl": "Tl_d",
        "Pb": "Pb_d",
        "Bi": "Bi_d",
        "Po": "Po_d",
        "At": "At",
        "Rn": "Rn",
        "Fr": "Fr_sv",
        "Ra": "Ra_sv",
        "Ac": "Ac",
        "Th": "Th",
        "Pa": "Pa",
        "U": "U",
        "Np": "Np",
        "Pu": "Pu",
        "Am": "Am",
        "Cm": "Cm",
        "Cf": "Cf",
    },
    "potpaw_PBE.64_GW": {
        "H": "H_GW",
        "He": "He_GW",
        "Li": "Li_sv_GW",
        "Be": "Be_sv_GW",
        "B": "B_GW",
        "C": "C_GW",
        "N": "N_GW",
        "O": "O_GW",
        "F": "F_GW",
        "Ne": "Ne_GW",
        "Na": "Na_sv_GW",
        "Mg": "Mg_sv_GW",
        "Al": "Al_GW",
        "Si": "Si_GW",
        "P": "P_GW",
        "S": "S_GW",
        "Cl": "Cl_GW",
        "Ar": "Ar_GW",
        "K": "K_sv_GW",
        "Ca": "Ca_sv_GW",
        "Sc": "Sc_sv_GW",
        "Ti": "Ti_sv_GW",
        "V": "V_sv_GW",
        "Cr": "Cr_sv_GW",
        "Mn": "Mn_sv_GW",
        "Fe": "Fe_sv_GW",
        "Co": "Co_sv_GW",
        "Ni": "Ni_sv_GW",
        "Cu": "Cu_sv_GW",
        "Zn": "Zn_sv_GW",
        "Ga": "Ga_d_GW",
        "Ge": "Ge_d_GW",
        "As": "As_GW",
        "Se": "Se_GW",
        "Br": "Br_GW",
        "Kr": "Kr_GW",
        "Rb": "Rb_sv_GW",
        "Sr": "Sr_sv_GW",
        "Y": "Y_sv_GW",
        "Zr": "Zr_sv_GW",
        "Nb": "Nb_sv_GW",
        "Mo": "Mo_sv_GW",
        "Tc": "Tc_sv_GW",
        "Ru": "Ru_sv_GW",
        "Rh": "Rh_sv_GW",
        "Pd": "Pd_sv_GW",
        "Ag": "Ag_sv_GW",
        "Cd": "Cd_sv_GW",
        "In": "In_d_GW",
        "Sn": "Sn_d_GW",
        "Sb": "Sb_d_GW",
        "Te": "Te_GW",
        "I": "I_GW",
        "Xe": "Xe_GW",
        "Cs": "Cs_sv_GW",
        "Ba": "Ba_sv_GW",
        "La": "La_GW",
        "Ce": "Ce_GW",
        "Hf": "Hf_sv_GW",
        "Ta": "Ta_sv_GW",
        "W": "W_sv_GW",
        "Re": "Re_sv_GW",
        "Os": "Os_sv_GW",
        "Ir": "Ir_sv_GW",
        "Pt": "Pt_sv_GW",
        "Au": "Au_sv_GW",
        "Hg": "Hg_sv_GW",
        "Tl": "Tl_d_GW",
        "Pb": "Pb_d_GW",
        "Bi": "Bi_d_GW",
        "Po": "Po_d_GW",
        "At": "At_d_GW",
        "Rn": "Rn_d_GW",
    },
    "potpaw_LDA.64_GW": {
        "H": "H_GW",
        "He": "He_GW",
        "Li": "Li_sv_GW",
        "Be": "Be_sv_GW",
        "B": "B_GW",
        "C": "C_GW",
        "N": "N_GW",
        "O": "O_GW",
        "F": "F_GW",
        "Ne": "Ne_GW",
        "Na": "Na_sv_GW",
        "Mg": "Mg_sv_GW",
        "Al": "Al_GW",
        "Si": "Si_GW",
        "P": "P_GW",
        "S": "S_GW",
        "Cl": "Cl_GW",
        "Ar": "Ar_GW",
        "K": "K_sv_GW",
        "Ca": "Ca_sv_GW",
        "Sc": "Sc_sv_GW",
        "Ti": "Ti_sv_GW",
        "V": "V_sv_GW",
        "Cr": "Cr_sv_GW",
        "Mn": "Mn_sv_GW",
        "Fe": "Fe_sv_GW",
        "Co": "Co_sv_GW",
        "Ni": "Ni_sv_GW",
        "Cu": "Cu_sv_GW",
        "Zn": "Zn_sv_GW",
        "Ga": "Ga_d_GW",
        "Ge": "Ge_d_GW",
        "As": "As_GW",
        "Se": "Se_GW",
        "Br": "Br_GW",
        "Kr": "Kr_GW",
        "Rb": "Rb_sv_GW",
        "Sr": "Sr_sv_GW",
        "Y": "Y_sv_GW",
        "Zr": "Zr_sv_GW",
        "Nb": "Nb_sv_GW",
        "Mo": "Mo_sv_GW",
        "Tc": "Tc_sv_GW",
        "Ru": "Ru_sv_GW",
        "Rh": "Rh_sv_GW",
        "Pd": "Pd_sv_GW",
        "Ag": "Ag_sv_GW",
        "Cd": "Cd_sv_GW",
        "In": "In_d_GW",
        "Sn": "Sn_d_GW",
        "Sb": "Sb_d_GW",
        "Te": "Te_GW",
        "I": "I_GW",
        "Xe": "Xe_GW",
        "Cs": "Cs_sv_GW",
        "Ba": "Ba_sv_GW",
        "La": "La_GW",
        "Ce": "Ce_GW",
        "Hf": "Hf_sv_GW",
        "Ta": "Ta_sv_GW",
        "W": "W_sv_GW",
        "Re": "Re_sv_GW",
        "Os": "Os_sv_GW",
        "Ir": "Ir_sv_GW",
        "Pt": "Pt_sv_GW",
        "Au": "Au_sv_GW",
        "Hg": "Hg_sv_GW",
        "Tl": "Tl_d_GW",
        "Pb": "Pb_d_GW",
        "Bi": "Bi_d_GW",
        "Po": "Po_d_GW",
        "At": "At_d_GW",
        "Rn": "Rn_d_GW",
    },
    "potpaw_PBE.54": {
        "H": "H",
        "He": "He",
        "Li": "Li_sv",
        "Be": "Be",
        "B": "B",
        "C": "C",
        "N": "N",
        "O": "O",
        "F": "F",
        "Ne": "Ne",
        "Na": "Na_pv",
        "Mg": "Mg",
        "Al": "Al",
        "Si": "Si",
        "P": "P",
        "S": "S",
        "Cl": "Cl",
        "Ar": "Ar",
        "K": "K_sv",
        "Ca": "Ca_sv",
        "Sc": "Sc_sv",
        "Ti": "Ti_sv",
        "V": "V_sv",
        "Cr": "Cr_pv",
        "Mn": "Mn_pv",
        "Fe": "Fe",
        "Co": "Co",
        "Ni": "Ni",
        "Cu": "Cu",
        "Zn": "Zn",
        "Ga": "Ga_d",
        "Ge": "Ge_d",
        "As": "As",
        "Se": "Se",
        "Br": "Br",
        "Kr": "Kr",
        "Rb": "Rb_sv",
        "Sr": "Sr_sv",
        "Y": "Y_sv",
        "Zr": "Zr_sv",
        "Nb": "Nb_sv",
        "Mo": "Mo_sv",
        "Tc": "Tc_pv",
        "Ru": "Ru_pv",
        "Rh": "Rh_pv",
        "Pd": "Pd",
        "Ag": "Ag",
        "Cd": "Cd",
        "In": "In_d",
        "Sn": "Sn_d",
        "Sb": "Sb",
        "Te": "Te",
        "I": "I",
        "Xe": "Xe",
        "Cs": "Cs_sv",
        "Ba": "Ba_sv",
        "La": "La",
        "Ce": "Ce",
        "Pr": "Pr_3",
        "Nd": "Nd_3",
        "Pm": "Pm_3",
        "Sm": "Sm_3",
        "Eu": "Eu_2",
        "Gd": "Gd_3",
        "Tb": "Tb_3",
        "Dy": "Dy_3",
        "Ho": "Ho_3",
        "Er": "Er_3",
        "Tm": "Tm_3",
        "Yb": "Yb_2",
        "Lu": "Lu_3",
        "Hf": "Hf_pv",
        "Ta": "Ta_pv",
        "W": "W_sv",
        "Re": "Re",
        "Os": "Os",
        "Ir": "Ir",
        "Pt": "Pt",
        "Au": "Au",
        "Hg": "Hg",
        "Tl": "Tl_d",
        "Pb": "Pb_d",
        "Bi": "Bi_d",
        "Po": "Po_d",
        "At": "At",
        "Rn": "Rn",
        "Fr": "Fr_sv",
        "Ra": "Ra_sv",
        "Ac": "Ac",
        "Th": "Th",
        "Pa": "Pa",
        "U": "U",
        "Np": "Np",
        "Pu": "Pu",
        "Am": "Am",
        "Cm": "Cm",
        "Cf": "Cf",
    },
    "potpaw_LDA.54": {
        "H": "H",
        "He": "He",
        "Li": "Li_sv",
        "Be": "Be",
        "B": "B",
        "C": "C",
        "N": "N",
        "O": "O",
        "F": "F",
        "Ne": "Ne",
        "Na": "Na_pv",
        "Mg": "Mg",
        "Al": "Al",
        "Si": "Si",
        "P": "P",
        "S": "S",
        "Cl": "Cl",
        "Ar": "Ar",
        "K": "K_sv",
        "Ca": "Ca_sv",
        "Sc": "Sc_sv",
        "Ti": "Ti_sv",
        "V": "V_sv",
        "Cr": "Cr_pv",
        "Mn": "Mn_pv",
        "Fe": "Fe",
        "Co": "Co",
        "Ni": "Ni",
        "Cu": "Cu",
        "Zn": "Zn",
        "Ga": "Ga_d",
        "Ge": "Ge_d",
        "As": "As",
        "Se": "Se",
        "Br": "Br",
        "Kr": "Kr",
        "Rb": "Rb_sv",
        "Sr": "Sr_sv",
        "Y": "Y_sv",
        "Zr": "Zr_sv",
        "Nb": "Nb_sv",
        "Mo": "Mo_sv",
        "Tc": "Tc_pv",
        "Ru": "Ru_pv",
        "Rh": "Rh_pv",
        "Pd": "Pd",
        "Ag": "Ag",
        "Cd": "Cd",
        "In": "In_d",
        "Sn": "Sn_d",
        "Sb": "Sb",
        "Te": "Te",
        "I": "I",
        "Xe": "Xe",
        "Cs": "Cs_sv",
        "Ba": "Ba_sv",
        "La": "La",
        "Ce": "Ce",
        "Hf": "Hf_pv",
        "Ta": "Ta_pv",
        "W": "W_sv",
        "Re": "Re",
        "Os": "Os",
        "Ir": "Ir",
        "Pt": "Pt",
        "Au": "Au",
        "Hg": "Hg",
        "Tl": "Tl_d",
        "Pb": "Pb_d",
        "Bi": "Bi_d",
        "Po": "Po_d",
        "At": "At",
        "Rn": "Rn",
        "Fr": "Fr_sv",
        "Ra": "Ra_sv",
        "Ac": "Ac",
        "Th": "Th",
        "Pa": "Pa",
        "U": "U",
        "Np": "Np",
        "Pu": "Pu",
        "Am": "Am",
        "Cm": "Cm",
    },
    "potpaw_PBE.54_GW": {
        "H": "H_GW",
        "He": "He_GW",
        "Li": "Li_sv_GW",
        "Be": "Be_sv_GW",
        "B": "B_GW",
        "C": "C_GW",
        "N": "N_GW",
        "O": "O_GW",
        "F": "F_GW",
        "Ne": "Ne_GW",
        "Na": "Na_sv_GW",
        "Mg": "Mg_sv_GW",
        "Al": "Al_GW",
        "Si": "Si_GW",
        "P": "P_GW",
        "S": "S_GW",
        "Cl": "Cl_GW",
        "Ar": "Ar_GW",
        "K": "K_sv_GW",
        "Ca": "Ca_sv_GW",
        "Sc": "Sc_sv_GW",
        "Ti": "Ti_sv_GW",
        "V": "V_sv_GW",
        "Cr": "Cr_sv_GW",
        "Mn": "Mn_sv_GW",
        "Fe": "Fe_sv_GW",
        "Co": "Co_sv_GW",
        "Ni": "Ni_sv_GW",
        "Cu": "Cu_sv_GW",
        "Zn": "Zn_sv_GW",
        "Ga": "Ga_d_GW",
        "Ge": "Ge_d_GW",
        "As": "As_GW",
        "Se": "Se_GW",
        "Br": "Br_GW",
        "Kr": "Kr_GW",
        "Rb": "Rb_sv_GW",
        "Sr": "Sr_sv_GW",
        "Y": "Y_sv_GW",
        "Zr": "Zr_sv_GW",
        "Nb": "Nb_sv_GW",
        "Mo": "Mo_sv_GW",
        "Tc": "Tc_sv_GW",
        "Ru": "Ru_sv_GW",
        "Rh": "Rh_sv_GW",
        "Pd": "Pd_sv_GW",
        "Ag": "Ag_sv_GW",
        "Cd": "Cd_sv_GW",
        "In": "In_d_GW",
        "Sn": "Sn_d_GW",
        "Sb": "Sb_d_GW",
        "Te": "Te_GW",
        "I": "I_GW",
        "Xe": "Xe_GW",
        "Cs": "Cs_sv_GW",
        "Ba": "Ba_sv_GW",
        "La": "La_GW",
        "Ce": "Ce_GW",
        "Hf": "Hf_sv_GW",
        "Ta": "Ta_sv_GW",
        "W": "W_sv_GW",
        "Re": "Re_sv_GW",
        "Os": "Os_sv_GW",
        "Ir": "Ir_sv_GW",
        "Pt": "Pt_sv_GW",
        "Au": "Au_sv_GW",
        "Hg": "Hg_sv_GW",
        "Tl": "Tl_d_GW",
        "Pb": "Pb_d_GW",
        "Bi": "Bi_d_GW",
        "Po": "Po_d_GW",
        "At": "At_d_GW",
        "Rn": "Rn_d_GW",
    },
    "potpaw_LDA.54_GW": {
        "H": "H_GW",
        "He": "He_GW",
        "Li": "Li_sv_GW",
        "Be": "Be_sv_GW",
        "B": "B_GW",
        "C": "C_GW",
        "N": "N_GW",
        "O": "O_GW",
        "F": "F_GW",
        "Ne": "Ne_GW",
        "Na": "Na_sv_GW",
        "Mg": "Mg_sv_GW",
        "Al": "Al_GW",
        "Si": "Si_GW",
        "P": "P_GW",
        "S": "S_GW",
        "Cl": "Cl_GW",
        "Ar": "Ar_GW",
        "K": "K_sv_GW",
        "Ca": "Ca_sv_GW",
        "Sc": "Sc_sv_GW",
        "Ti": "Ti_sv_GW",
        "V": "V_sv_GW",
        "Cr": "Cr_sv_GW",
        "Mn": "Mn_sv_GW",
        "Fe": "Fe_sv_GW",
        "Co": "Co_sv_GW",
        "Ni": "Ni_sv_GW",
        "Cu": "Cu_sv_GW",
        "Zn": "Zn_sv_GW",
        "Ga": "Ga_d_GW",
        "Ge": "Ge_d_GW",
        "As": "As_GW",
        "Se": "Se_GW",
        "Br": "Br_GW",
        "Kr": "Kr_GW",
        "Rb": "Rb_sv_GW",
        "Sr": "Sr_sv_GW",
        "Y": "Y_sv_GW",
        "Zr": "Zr_sv_GW",
        "Nb": "Nb_sv_GW",
        "Mo": "Mo_sv_GW",
        "Tc": "Tc_sv_GW",
        "Ru": "Ru_sv_GW",
        "Rh": "Rh_sv_GW",
        "Pd": "Pd_sv_GW",
        "Ag": "Ag_sv_GW",
        "Cd": "Cd_sv_GW",
        "In": "In_d_GW",
        "Sn": "Sn_d_GW",
        "Sb": "Sb_d_GW",
        "Te": "Te_GW",
        "I": "I_GW",
        "Xe": "Xe_GW",
        "Cs": "Cs_sv_GW",
        "Ba": "Ba_sv_GW",
        "La": "La_GW",
        "Ce": "Ce_GW",
        "Hf": "Hf_sv_GW",
        "Ta": "Ta_sv_GW",
        "W": "W_sv_GW",
        "Re": "Re_sv_GW",
        "Os": "Os_sv_GW",
        "Ir": "Ir_sv_GW",
        "Pt": "Pt_sv_GW",
        "Au": "Au_sv_GW",
        "Hg": "Hg_sv_GW",
        "Tl": "Tl_d_GW",
        "Pb": "Pb_d_GW",
        "Bi": "Bi_d_GW",
        "Po": "Po_d_GW",
        "At": "At_d_GW",
        "Rn": "Rn_d_GW",
    },
}

element_layout = [
    [
        "H",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "He",
    ],  # noqa: E501, E201
    [
        "Li",
        "Be",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
    ],  # noqa: E501
    [
        "Na",
        "Mg",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
    ],  # noqa: E501
    [
        "K",
        "Ca",
        "",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
    ],  # noqa: E501, E201
    [
        "Rb",
        "Sr",
        "",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
    ],  # noqa: E501
    [
        "Cs",
        "Ba",
        "",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
    ],  # noqa: E501
    [
        "Fr",
        "Ra",
        "",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ],  # noqa: E501
    [],
    [
        "",
        "",
        "",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
    ],  # noqa: E501, E201
    [
        "",
        "",
        "",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
    ],  # noqa: E501, E201
]  # yapf: disable


class TkEnergy(seamm.TkNode):
    """
    The graphical part of a Energy step in a flowchart.

    Attributes
    ----------
    tk_flowchart : TkFlowchart = None
        The flowchart that we belong to.
    node : Node = None
        The corresponding node of the non-graphical flowchart
    canvas: tkCanvas = None
        The Tk Canvas to draw on
    dialog : Dialog
        The Pmw dialog object
    x : int = None
        The x-coordinate of the center of the picture of the node
    y : int = None
        The y-coordinate of the center of the picture of the node
    w : int = 200
        The width in pixels of the picture of the node
    h : int = 50
        The height in pixels of the picture of the node
    self[widget] : dict
        A dictionary of tk widgets built using the information
        contained in Energy_parameters.py

    See Also
    --------
    Energy, TkEnergy,
    EnergyParameters,
    """

    def __init__(
        self,
        tk_flowchart=None,
        node=None,
        canvas=None,
        x=None,
        y=None,
        w=200,
        h=50,
    ):
        """
        Initialize a graphical node.

        Parameters
        ----------
        tk_flowchart: Tk_Flowchart
            The graphical flowchart that we are in.
        node: Node
            The non-graphical node for this step.
        namespace: str
            The stevedore namespace for finding sub-nodes.
        canvas: Canvas
           The Tk canvas to draw on.
        x: float
            The x position of the nodes center on the canvas.
        y: float
            The y position of the nodes cetner on the canvas.
        w: float
            The nodes graphical width, in pixels.
        h: float
            The nodes graphical height, in pixels.

        Returns
        -------
        None
        """
        self.dialog = None
        self.available = {}  # The available potentials for the current set
        self.current_potential_set = None
        self.default_potential = {}

        super().__init__(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=canvas,
            x=x,
            y=y,
            w=w,
            h=h,
        )

        # Make sure we have the information about the PAW potentials
        directory = Path("~/SEAMM/Parameters/VASP").expanduser()
        index = directory / "index.json"
        if index.exists():
            with index.open() as fd:
                self.potential_metadata = json.load(fd)
        else:
            self.potential_metadata = self.node.parent.catalog_potentials()
            with index.open("w") as fd:
                json.dump(
                    self.potential_metadata,
                    fd,
                    indent=4,
                    sort_keys=True,
                    cls=CompactJSONEncoder,
                )
            print(f"Wrote the index to {index}")

    def create_dialog(self, title="Energy"):
        """
        Create the dialog. A set of widgets will be chosen by default
        based on what is specified in the Energy_parameters
        module.

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.reset_dialog
        """

        # Shortcut for parameters
        P = self.node.parameters

        frame = super().create_dialog(title=title, default_number_values=1)

        # Make scrollable in case contents too large
        self["scrolled frame"] = sw.ScrolledFrame(frame)
        self["scrolled frame"].grid(row=0, column=0, sticky=tk.NSEW)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        main_frame = self["main frame"] = self["scrolled frame"].interior()

        # Then create the widgets
        for key in ("input only", "calculate stress"):
            self[key] = P[key].widget(main_frame)

        # Frame to isolate widgets
        e_frame = self["model frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Electronic Structure Definition",
            labelanchor="n",
            padding=10,
        )

        # Then create the widgets
        for key in ("model", "submodel", "spin polarization", "nonspherical PAW"):
            self[key] = P[key].widget(e_frame)
            try:
                self[key].configure(width=45)
            except Exception:
                pass

        for key in ("model", "submodel"):
            self[key].bind("<<ComboboxSelected>>", self.reset_model_frame)
            self[key].bind("<Return>", self.reset_model_frame)
            self[key].bind("<FocusOut>", self.reset_model_frame)

        # Frame for the control of the electronic minimization
        scf_frame = self["scf frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Electronic Minimization",
            labelanchor="n",
            padding=10,
        )

        # Then create the widgets and place them
        row = 0
        widgets = []
        for key in ("electronic method", "nelm", "ediff"):
            w = self[key] = P[key].widget(scf_frame)
            w.grid(row=row, column=0, sticky="ew")
            row += 1
            widgets.append(w)
        sw.align_labels(widgets, sticky="e")

        # The k-space integration
        k_frame = self["k-space frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="k-space integration (for periodic systems)",
            labelanchor="n",
            padding=10,
        )
        for key in vasp_step.EnergyParameters.kspace_parameters:
            self[key] = P[key].widget(k_frame)

        for key in ("na", "nb", "nc"):
            self[key].entry.configure(width=4)

        for key in ("k-grid method", "occupation type"):
            self[key].bind("<<ComboboxSelected>>", self.reset_kspace_frame)
            self[key].bind("<Return>", self.reset_kspace_frame)
            self[key].bind("<FocusOut>", self.reset_kspace_frame)

        # Frame for the performance parameters
        performance_frame = self["performance frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Performance Parameters",
            labelanchor="n",
            padding=10,
        )

        # Then create the widgets and place them
        row = 0
        widgets = []
        for key in ("ncore", "kpar", "lplane", "lreal"):
            w = self[key] = P[key].widget(performance_frame)
            w.grid(row=row, column=0, sticky="ew")
            row += 1
            widgets.append(w)
        sw.align_labels(widgets, sticky="e")

        # and lay them out
        self.reset_dialog()

        self.create_potentials_tab()

    def create_potentials_tab(self):
        """Add a tab with a periodic table for setting the potentials."""
        # Shortcut for parameters
        P = self.node.parameters

        notebook = self["notebook"]
        potentials_frame = ttk.Frame(notebook)
        self["potentials frame"] = potentials_frame
        notebook.insert(
            self["results frame"],
            potentials_frame,
            text="Potentials",
            sticky="nsew",
        )

        row = 0

        widgets = []
        for key in ("set of potentials", "plane-wave cutoff"):
            self[key] = P[key].widget(potentials_frame)
            self[key].grid(row=row, column=0, sticky="ew")
            row += 1
            widgets.append(self[key])
        # Align the labels
        sw.align_labels(widgets, sticky="e")
        for key in ("set of potentials",):
            self[key].bind("<<ComboboxSelected>>", self._change_set)
            self[key].bind("<Return>", self._change_set)
            self[key].bind("<FocusOut>", self._change_set)

        # Display the max enmax next to the plane-wave cutoff
        for key in ("enmax",):
            w = self[key] = P[key].widget(potentials_frame)
            w.grid(row=1, column=1, sticky="w")
            w.config(state="readonly")

        # Put the possible potential sets into the widget
        metadata = self.potential_metadata
        sets = [*metadata.keys()]
        self.current_potential_set = self["set of potentials"].value
        self["set of potentials"].config(values=sets)
        if "GW" in P["subset"]:
            self.default_potential = default_potentials[
                self.current_potential_set + "_GW"
            ]
        else:
            self.default_potential = default_potentials[self.current_potential_set]

        # This is used for the radiobutton to control what subset is show/used
        var = self.tk_var["subset"] = tk.StringVar()
        var.set(P["subset"])

        # And make a scrollable periodic table area for the potentials
        self["scrolled periodic table"] = sw.ScrolledFrame(potentials_frame)
        self["scrolled periodic table"].grid(
            row=row, column=0, columnspan=2, sticky=tk.NSEW
        )
        potentials_frame.rowconfigure(row, weight=1)
        potentials_frame.columnconfigure(1, weight=1)
        row += 1

        frame = self["potentials"] = self["scrolled periodic table"].interior()

        # Create the widgets in the periodic table layout
        current = self.node.parameters["potentials"].value
        self.find_available_potentials()

        for _row, elements in enumerate(element_layout):
            for _col, element in enumerate(elements):
                if element != "":
                    if element in self.available:
                        w = self._widget[element] = ttk.Combobox(
                            frame,
                            values=[f"_{element}_", *self.available[element]],
                            width=6,
                        )
                        if element in current:
                            w.set(current[element])
                        else:
                            w.set(self.default_potential[element])
                    else:
                        w = self._widget[element] = ttk.Combobox(
                            frame,
                            width=5,
                        )
                        w.set(f"_{element}_")
                        w.config(state="disabled")
                    w.grid(row=_row, column=_col, sticky=tk.EW)
                    w.bind("<<ComboboxSelected>>", self.update_Emax)
                    w.bind("<Return>", self.update_Emax)
                    w.bind("<FocusOut>", self.update_Emax)

        self.update_Emax()

        # Radio button to allow selecting standard and GW potentials.
        for text in (
            "Show all potentials",
            "Show standard potentials",
            "Show GW potentials",
        ):
            w = self._widget[text] = ttk.Radiobutton(
                potentials_frame,
                text=text,
                value=text,
                variable=self.tk_var["subset"],
                command=self._handle_show_potentials,
            )
            w.grid(row=row, column=0, sticky="w")
            row += 1

        w = self._widget["default potentials"] = ttk.Button(
            potentials_frame,
            text="Reset to default potentials",
            command=self._set_to_default_potentials,
        )
        w.grid(row=row, column=0, sticky="w")
        row += 1

        w = self._widget["clear all potentials"] = ttk.Button(
            potentials_frame,
            text="Clear all potentials",
            command=self._clear_potentials,
        )
        w.grid(row=row, column=0, sticky="w")
        row += 1

        frame.grid_columnconfigure(2, minsize=30)
        frame.grid_rowconfigure(7, minsize=30)

    def _change_set(self, *args):
        """Handle changing the set of potentials to use."""
        self.current_potential_set = self["set of potentials"].get()
        self._handle_show_potentials()

    def _handle_show_potentials(self, current=None):
        """Show or hide the standard or GW potentials."""
        subset = self.tk_var["subset"].get()
        self.find_available_potentials()
        if "GW" in subset:
            self.default_potential = default_potentials[
                self.current_potential_set + "_GW"
            ]
        else:
            self.default_potential = default_potentials[self.current_potential_set]
        if current is None:
            current = self.get_current_potentials()
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        self._widget[element].config(
                            values=[f"_{element}_", *self.available[element]],
                            state="!disabled",
                        )
                        if element in current:
                            if current[element] in self.available[element]:
                                self._widget[element].set(current[element])
                            else:
                                self._widget[element].set(
                                    self.default_potential[element]
                                )
                        else:
                            self._widget[element].set(f"_{element}_")
                    else:
                        self._widget[element].config(state="!disabled")
                        self._widget[element].set(f"_{element}_")
                        self._widget[element].config(
                            values=[f"_{element}_"], state="disabled"
                        )
        self.update_Emax()

    def _set_to_default_potentials(self):
        """Reset to the default potentials."""
        self.find_available_potentials()
        subset = self.tk_var["subset"].get()
        if "GW" in subset:
            self.default_potential = default_potentials[
                self.current_potential_set + "_GW"
            ]
        else:
            self.default_potential = default_potentials[self.current_potential_set]
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        self._widget[element].set(self.default_potential[element])
                    else:
                        self._widget[element].config(state="!disabled")
                        self._widget[element].set(f"_{element}_")
                        self._widget[element].config(state="disabled")
        self.update_Emax()

    def _clear_potentials(self):
        """Reset to the no potentials."""
        for elements in element_layout:
            for element in elements:
                if element != "":
                    self._widget[element].set(f"_{element}_")
        self.update_Emax()

    def find_available_potentials(self):
        potential_data = self.potential_metadata[self.current_potential_set]
        self.available = {}
        subset = self.tk_var["subset"].get()
        for potential, data in potential_data.items():
            if "GW" in subset:
                if "_GW" not in potential:
                    continue
            elif "all" not in subset:
                if "_GW" in potential:
                    continue
            el = potential.split("_")[0]
            if el in self.available:
                self.available[el].append(potential)
            else:
                self.available[el] = [potential]
        for el, tmp in self.available.items():
            self.available[el] = sorted(tmp)

    def get_current_potentials(self):
        """Get the currently selected potentials from the widgets."""
        result = {}
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        tmp = self._widget[element].get()
                        if tmp != "" and not tmp.startswith("_"):
                            result[element] = tmp
        return result

    def get_Emax(self):
        """Get the maximum ENMAX for the currently slsected potentials"""
        potential_data = self.potential_metadata[self.current_potential_set]
        Emax = 0
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        potential = self._widget[element].get()
                        if potential.startswith("_") or potential == "":
                            pass
                        else:
                            Emax = max(Emax, float(potential_data[potential]["Emax"]))
        return Emax

    def update_Emax(self, event=None):
        """Update the value of Emax in the display."""
        self["enmax"].config(state="!readonly")
        self["enmax"].set(self.get_Emax(), unit_string="eV")
        self["enmax"].config(state="readonly")

    def reset_dialog(self, widget=None):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        Energy_parameter.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.create_dialog
        """

        # Remove any widgets previously packed
        frame = self["main frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # Shortcut for parameters
        P = self.node.parameters

        # keep track of the row in a variable, so that the layout is flexible
        # if e.g. rows are skipped to control such as "method" here
        row = 0

        # Whether to just write input
        self["input only"].grid(row=row, column=0, sticky=tk.W)
        row += 1

        # The model for the calculation
        self["model frame"].grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")
        row += 1
        self["k-space frame"].grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")
        row += 1
        self["scf frame"].grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")
        row += 1
        self["performance frame"].grid(
            row=row, column=0, columnspan=2, pady=5, sticky="ew"
        )
        row += 1

        # Other parameters for a single-point calculation
        if type(self) is vasp_step.TkEnergy:
            keys = ["calculate stress"]
            for key in keys:
                self[key].grid(row=row, column=0, columnspan=2, pady=10, sticky="w")
                row += 1

        # Layout the energy widgets
        self.reset_model_frame()
        self.reset_kspace_frame()

        # Setup the results if there are any
        have_results = (
            "results" in self.node.metadata and len(self.node.metadata["results"]) > 0
        )
        if have_results and "results" in P:
            self.setup_results()

        return row

    def reset_model_frame(self, widget=None):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        Energy_parameter.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.create_dialog
        """

        # Remove any widgets previously packed
        frame = self["model frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # Shortcut for parameters
        P = self.node.parameters

        # keep track of the row in a variable, so that the layout is flexible
        # if e.g. rows are skipped to control such as "method" here
        row = 0

        # The model for the calculation
        keys = ["model", "submodel", "spin polarization", "nonspherical PAW"]
        model = self["model"].get()

        # And what models have available sublevels?
        model_data = self.node.metadata["computational models"][
            "Density Functional Theory (DFT)"
        ]["models"]
        models = []
        for tmp in model_data.keys():
            models.append(tmp)
        w = self["model"]
        if w.combobox.cget("values") != models:
            w.config(values=models)
            if not self.is_expr(model) and model not in models and len(models) > 0:
                w.set(models[0])
            else:
                w.set(model)

        # Fill out the possible submodels
        submodel = self["submodel"].get()
        choices = model_data[model]["parameterizations"]
        submodels = [*choices.keys()]

        w = self["submodel"]
        if w.combobox.cget("values") != submodels:
            w.config(values=submodels)
            if (
                not self.is_expr(submodel)
                and submodel not in submodels
                and len(submodels) > 0
            ):
                w.set(submodels[0])
            else:
                w.set(submodel)

        # And the basis
        widgets = []

        for key in keys:
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
            row += 1

        # Align the labels
        sw.align_labels(widgets, sticky=tk.E)

        # Setup the results if there are any
        have_results = (
            "results" in self.node.metadata and len(self.node.metadata["results"]) > 0
        )
        if have_results and "results" in P:
            self.setup_results()

    def reset_kspace_frame(self, widget=None):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        Energy_parameter.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.create_dialog
        """

        # Remove any widgets previously packed
        frame = self["k-space frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # keep track of the row in a variable, so that the layout is flexible
        # if e.g. rows are skipped to control such as "method" here
        row = 0

        # The model for the calculation
        key = "k-grid method"
        method = self[key].get()
        self[key].grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        if "explicit" in method:
            for col, key in enumerate(("na", "nb", "nc"), start=1):
                self[key].grid(row=row, column=col, sticky=tk.EW)
            row += 1
        else:
            widgets = []
            for key in ("k-spacing", "odd grid", "centering"):
                self[key].grid(row=row, column=1, columnspan=3, sticky=tk.EW)
                widgets.append(self[key])
                row += 1
            sw.align_labels(widgets, sticky=tk.E)
        frame.columnconfigure(0, minsize=30)

        key = "occupation type"
        smearing = self[key].get()
        self[key].grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        if "Methfessel-Paxton" in smearing:
            key = "Methfessel-Paxton order"
            self[key].grid(row=row, column=1, columnspan=3, sticky=tk.W)
            row += 1

        if "without smearing" not in smearing:
            key = "smearing width"
            self[key].grid(row=row, column=1, columnspan=3, sticky=tk.W)
            row += 1

    def right_click(self, event):
        """
        Handles the right click event on the node.

        Parameters
        ----------
        event : Tk Event

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.edit
        """

        super().right_click(event)
        self.popup_menu.add_command(label="Edit..", command=self.edit)

        self.popup_menu.tk_popup(event.x_root, event.y_root, 0)

    def fit_dialog(self):
        """Make the dialog fill the window."""
        screen_w = self.dialog.winfo_screenwidth()
        screen_h = self.dialog.winfo_screenheight()
        w = int(1.0 * screen_w)
        h = int(1.0 * screen_h)
        x = 0
        y = int(0.0 * screen_h / 2)

        self.dialog.geometry(f"{w}x{h}+{x}+{y}")

    def handle_dialog(self, result):
        """Handle closing the dialog.

        Parameters
        ----------
        result : str
            The button that was pressed to close the dialog, or None if the x dialog
            close button was pressed.
        """
        if result is None or result == "Cancel":
            # Reset the potentials to their original values
            P = self.node.parameters
            subset = P["subset"]
            self.tk_var["subset"].set(subset)
            for key in ("set of potentials", "plane-wave cutoff"):
                self[key].set(P[key])
            self.current_potential_set = P["set of potentials"].value
            if "GW" in subset:
                self.default_potential = default_potentials[
                    self.current_potential_set + "_GW"
                ]
            else:
                self.default_potential = default_potentials[self.current_potential_set]
            current = self.node.parameters["potentials"].value
            self._handle_show_potentials(current)
        elif result == "OK":
            self.node.parameters["potentials"].value = self.get_current_potentials()

        # And let the superclass do the rest
        super().handle_dialog(result)
