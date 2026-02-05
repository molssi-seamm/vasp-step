# -*- coding: utf-8 -*-

"""This file contains metadata describing the results from VASP"""

metadata = {}

"""Description of the computational models for VASP.

Hamiltonians, approximations, and basis set or parameterizations,
only if appropriate for this code. For example::

    metadata["computational models"] = {
        "Hartree-Fock": {
            "models": {
                "PM7": {
                    "parameterizations": {
                        "PM7": {
                            "elements": "1-60,62-83",
                            "periodic": True,
                            "reactions": True,
                            "optimization": True,
                            "code": "mopac",
                        },
                        "PM7-TS": {
                            "elements": "1-60,62-83",
                            "periodic": True,
                            "reactions": True,
                            "optimization": False,
                            "code": "mopac",
                        },
                    },
                },
            },
        },
    }
"""
metadata["computational models"] = {
    "Density Functional Theory (DFT)": {
        "models": {
            "Local-density approximation (LDA)": {
                "parameterizations": {
                    "VWN5 : Vosko-Wilk-Nusair": {
                        "description": (
                            "Slater exchange + LDA of Vosko, Wilk, and Nusair 1980. "
                            "(VWN5)",
                        ),
                        "keywords": {"GGA": "VW"},
                    },
                    "PW92 : Perdew-Wang": {
                        "description": (
                            "Homogeneous electron gas based on Ceperley and Alder "
                            "as parameterized by Perdew and Wang 1992. "
                            "Recommended LDA parameterization."
                        ),
                        "keywords": {"GGA": "PW92"},
                    },
                    "PZ-LDA : Perdew-Zunger": {
                        "description": (
                            "Homogeneous electron gas based on Ceperley and Alder [42],"
                            " as parameterized by Perdew and Zunger 1981 [184]"
                        ),
                        "keywords": {"GGA": "PZ"},
                    },
                }
            },
            "Generalized-gradient approximations (GGA)": {
                "parameterizations": {
                    "PBE : Perdew, Burke and Ernzerhof": {
                        "description": "GGA of Perdew, Burke and Ernzerhof 1997.",
                        "keywords": {"GGA": "PE"},
                    },
                    "PBE-D3BJ : PBE with D3(BJ) dispersion": {
                        "description": "GGA of Perdew, Burke and Ernzerhof 1997.",
                        "keywords": {"GGA": "PE", "IVDW": "12"},
                    },
                    "AM05 : Armiento and Mattsson": {
                        "description": (
                            "GGA functional designed to include surface effects in "
                            "self-consistent density functional theory, according to "
                            "Armiento and Mattsson [9]"
                        ),
                        "keywords": {"GGA": "AM"},
                    },
                    "PBEsol : modified PBE GGA": {
                        "description": "Modified PBE GGA according to Ref. [187].",
                        "keywords": {"GGA": "PS"},
                    },
                    "PBEsol-D3BJ : PBEsol with D3(BJ) dispersion": {
                        "description": "Modified PBE GGA according to Ref. [187].",
                        "keywords": {"GGA": "PS", "IVDW": "12"},
                    },
                    "RPBE : the revised PBE functional of Hammer et al.": {
                        "description": (
                            "The RPBE modified PBE functional according to Ref. [93]."
                        ),
                        "keywords": {"GGA": "RP"},
                    },
                    "RPBE-D3BJ : RPBE with D3(BJ) dispersion": {
                        "description": (
                            "The RPBE modified PBE functional according to Ref. [93]."
                        ),
                        "keywords": {"GGA": "RP", "IVDW": "12"},
                    },
                    "revPBE : the revised PBE functional of Zhang and Yang": {
                        "description": (
                            "The revPBE modified PBE GGA suggested in Ref. [247]."
                        ),
                        "keywords": {"GGA": "RE", "IVDW": "12"},
                    },
                    "revPBE-D3BJ : revPBE with D3(BJ) dispersion": {
                        "description": (
                            "The revPBE modified PBE GGA suggested in Ref. [247]."
                        ),
                        "keywords": {"GGA": "RE", "IVDW": "12"},
                    },
                    "PW91 : Perdew-Wang 1991 GGA": {
                        "description": (
                            "GGA according to Perdew and Wang, usually referred to as "
                            "'Perdew-Wang 1991 GGA'. This GGA is most accessibly "
                            "described in Reference 26 and 27 of Ref. [182]. Note "
                            "that the often mis-quoted reference [183] does not(!) "
                            "describe the Perdew-Wang GGA but instead only the "
                            "correlation part of the local-density approximation "
                            "described above."
                        ),
                        "keywords": {"GGA": "91"},
                    },
                },
            },
            "Meta-generalized gradient approximations (meta-GGA)": {
                "parameterizations": {
                    "M06-l : Truhlar’s optimized meta-GGA": {
                        "description": (
                            "Truhlar’s optimized meta-GGA of the 'M06' suite of "
                            "functionals. [250]"
                        ),
                        "keywords": {"METAGGA": "M06L"},
                    },
                    "M06-l-D3BJ : M06-l with D3(BJ) dispersion": {
                        "description": (
                            "Truhlar’s optimized meta-GGA of the 'M06' suite of "
                            "functionals. [250]"
                        ),
                        "keywords": {"METAGGA": "M06L", "IVDW": "12"},
                    },
                    "revTPSS": {
                        "description": (
                            "Meta-GGA revTPSS functional of Ref. [185, 186]",
                        ),
                        "keywords": {"METAGGA": "RTPSS"},
                    },
                    "TPSS": {
                        "description": "Meta-GGA TPSS functional of Ref. [223]",
                        "keywords": {"METAGGA": "TPSS"},
                    },
                    "TPSS-D3BJ": {
                        "description": "TPSS with D3(BJ) dispersion",
                        "keywords": {"METAGGA": "TPSS", "IVDW": "12"},
                    },
                    "SCAN": {
                        "description": (
                            "'Strongly Constrained and Appropriately Normed Semilocal "
                            "Density Functional,' i.e., the SCAN meta-GGA functional "
                            "by Sun, Ruzsinszky, and Perdew.[219]"
                            " May possibly lead to numerical instabilities."
                            " rSCAN or rSCAN are more stable and should give similar"
                            " results."
                        ),
                        "keywords": {"METAGGA": "SCAN"},
                    },
                    "SCAN-D3BJ": {
                        "description": "SCAN with D3(BJ) dispersion",
                        "keywords": {"METAGGA": "SCAN", "IVDW": "12"},
                    },
                    "rSCAN": {
                        "description": (
                            "rSCAN is a regularized version of SCAN that is numerically"
                            " more stable."
                        ),
                        "keywords": {"METAGGA": "RSCAN"},
                    },
                    "r2SCAN": {
                        "description": (
                            "r2SCAN is a regularized version of SCAN that is"
                            "  numerically more stable."
                        ),
                        "keywords": {"METAGGA": "R2SCAN"},
                    },
                    "r2SCAN-D3BJ : r2SCAN with D3(BJ) dispersion": {
                        "description": (
                            "r2SCAN is a regularized version of SCAN that is"
                            " numerically more stable, with D3(BJ) long-range"
                            " dispersion correction parameters from ORCA."
                        ),
                        "keywords": {
                            "METAGGA": "R2SCAN",
                            "IVDW": "12",
                            "VDW_S8": "0.7898",
                            "VDW_A1": "0.4948",
                            "VDW_A2": "5.7308",
                        },
                    },
                    "r2SCAN-D4BJ : r2SCAN plus D4(BJ) dispersion corrections": {
                        "description": (
                            "r2SCAN is a regularized version of SCAN that is"
                            "  numerically more stable, with D4(BJ) long-range"
                            " dispersion correction parameters from ORCA."
                        ),
                        "keywords": {
                            "METAGGA": "R2SCAN",
                            "IVDW": "13",
                            "VDW_S6": "1.0000",
                            "VDW_S8": "0.60187490",
                            "VDW_A1": "0.51559235",
                            "VDW_A2": "5.77342911",
                        },
                    },
                },
            },
            "Hybrid functionals and Hartree-Fock": {
                "parameterizations": {
                    "B3LYP": {
                        "description": (
                            "'B3LYP' hybrid functional as allegedly implemented in the "
                            "VASP code (i.e., using the RPA version of the "
                            "Vosko-Wilk-Nusair local-density approximation, see Refs. "
                            "[233, 210] for details). Note that this is therefore not "
                            "exactly the same B3LYP as originally described by Becke "
                            "in 1993."
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "GGA": "B3",
                            "AEXX": "0.2",
                            "AGGAX": "0.72",
                            "AGGAC": "0.81",
                            "ALDAC": "0.19",
                        },
                    },
                    "HF : Hartree-Fock": {
                        "description": "Hartree-Fock exchange, only",
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "AEXX": "1.0",
                        },
                    },
                    "HSE03 : Heyd, Scuseria and Ernzerhof": {
                        "description": (
                            "Hybrid functional as used in Heyd, Scuseria and Ernzerhof "
                            "[105, 106]. In this functional, 25 % of the exchange "
                            "energy is split into a short-ranged, screened Hartree-Fock"
                            " part, and a PBE GGA-like functional for the longrange "
                            "part of exchange. The remaining 75 % exchange and full "
                            "correlation energy are treated as in PBE. As clarified in "
                            "Refs. [141, 106], two different screening parameters were "
                            "used in the short-range exchange part and longrange "
                            "exchange part of the original HSE functional, "
                            "respectively. The ’hse03’ functional in FHI-aims "
                            "reproduces these original values exactly."
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "GGA": "PE",
                            "HFSCREEN": "0.3",
                            "AEXX": "0.25",
                            "AGGAX": "0.75",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                    "HSE06 : Heyd, Scuseria and Ernzerhof": {
                        "description": (
                            "Hybrid functional according to Heyd, Scuseria and "
                            "Ernzerhof [105], following the naming convention "
                            "suggested in Ref. [141]. In this case, the additional "
                            "option value is needed, representing the single real, "
                            "positive screening parameter omega as clarified in Ref. "
                            "[141]. In this functional, 25 % of the exchange energy is "
                            "split into a short-ranged, screened Hartree-Fock part, "
                            "and a PBE GGA-like functional for the long-range part of "
                            "exchange. The remaining 75 % exchange and full "
                            "correlation energy are treated as in PBE."
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "GGA": "PE",
                            "HFSCREEN": "0.2",
                            "AEXX": "0.25",
                            "AGGAX": "0.75",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                    "HSE06sol : Heyd, Scuseria and Ernzerhof": {
                        "description": (
                            "Hybrid functional according to Heyd, Scuseria and "
                            "Ernzerhof [105], using PBEsol "
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "GGA": "PS",
                            "HFSCREEN": "0.2",
                            "AEXX": "0.25",
                            "AGGAX": "0.75",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                    "RSHXLDA": {
                        "description": (
                            "Hybrid functional of Gerber, Ángyán, Marsman, and Kresse,"
                            " Range separated hybrid density functional with long-range"
                            " Hartree-Fock exchange applied to solids (2007)"
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "LRHFCALC": ".TRUE.",
                            "GGA": "PZ",
                            "HFSCREEN": "0.75",
                            "AEXX": "1.0",
                            "AGGAX": "0.0",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                    "RSHXPBE": {
                        "description": (
                            "Hybrid functional of Gerber and Ángyán,"
                            " Hybrid Functional with Separated Range (2005)"
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "LRHFCALC": ".TRUE.",
                            "GGA": "PE",
                            "HFSCREEN": "0.91",
                            "AEXX": "1.0",
                            "AGGAX": "0.0",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                    "PBE0": {
                        "description": (
                            "PBE0 hybrid functional [1], mixing 75 % GGA exchange with "
                            "25 % Hartree-Fock exchange"
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "GGA": "PE",
                            "AEXX": "0.25",
                            "AGGAX": "0.75",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                    "PBEsol0": {
                        "gui": "recommended",
                        "description": (
                            "Hybrid functional in analogy to PBE0 [1], except that the "
                            "PBEsol [187] GGA functionals are used, mixing 75 % GGA "
                            "exchange with 25 % Hartree-Fock exchange."
                        ),
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "GGA": "PS",
                            "AEXX": "0.25",
                            "AGGAX": "0.75",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                },
            },
            "Hybrid meta generalized gradient approximation (Hybrid meta-GGA)": {
                "parameterizations": {
                    "SCAN0": {
                        "description": "SCAN hybrid meta-GGA",
                        "keywords": {
                            "LHFCALC": ".TRUE.",
                            "METAGGA": "SCAN",
                            "AEXX": "0.25",
                            "AGGAX": "0.75",
                            "AGGAC": "1.0",
                            "ALDAC": "1.0",
                        },
                    },
                },
            },
        }
    }
}

"""Description of the VASP keywords.

(Only needed if this code uses keywords)

Fields
------
description : str
    A human readable description of the keyword.
takes values : int (optional)
    Number of values the keyword takes. If missing the keyword takes no values.
default : str (optional)
    The default value(s) if the keyword takes values.
format : str (optional)
    How the keyword is formatted in the MOPAC input.

For example::
    metadata["keywords"] = {
        "0SCF": {
            "description": "Read in data, then stop",
        },
        "ALT_A": {
            "description": "In PDB files with alternative atoms, select atoms A",
            "takes values": 1,
            "default": "A",
            "format": "{}={}",
        },
    }
"""
metadata["keywords"] = {
    "ISEARCH": {
        "description": "0: use legacy linesearch; 1 for new.",
        "default": 1,
    },
    "ISTART": {
        "description": (
            "Whether to read WAVECAR: 0: from scratch; 1: constant E basis; 2: constant"
            " basis."
        ),
        "default": 0,
    },
    "LH5": {
        "description": "Use HDF5 files instead of text files",
        "default": ".True.",
    },
    "LWAVE": {
        "description": "Write the wavefunction to the WAVECAR text file",
        "default": ".False.",
    },
    "LWAVEH5": {
        "description": "Write the wavefunction to the vaspwave.h5 HDF5 file",
        "default": ".True.",
    },
    "LCHARG": {
        "description": "Write the charge density to the CHGCAR and CHG text files",
        "default": ".False.",
    },
    "LCHARGH5": {
        "description": "Write the charge density to the vaspwave.h5 HDF5 file",
        "default": ".True.",
    },
    "IBRION": {
        "description": "How to change the structure during the calculation",
        "default": "1",
    },
    "NSW": {
        "description": "The maximum number of ionic steps",
        "default": 100,
    },
    "NELM": {
        "description": "The maximum number of electronic self-consistency steps",
        "default": 60,
    },
    "NELMIN": {
        "description": (
            "The minimum number of electronic self-consistency steps. 2 is the default,"
            " 4-8 recommended for minimization and MD"
        ),
        "default": "2",
    },
    "NELMDL": {
        "description": (
            "The number of non-consistent electronic steps at beginning. "
            "negative for first ionic step, positive for all steps."
        ),
        "default": -5,
    },
    "EDIFF": {
        "description": "Convergence criterion for electronic self-consistency",
        "default": "1.0E-6",
    },
    "EDIFFG": {
        "description": (
            "Convergence criterion for optimizations: positive for energy (eV) or "
            "negative for forces [(eV/Å)^2]"
        ),
        "default": "-1.0E-06",
    },
    "EFERMI": {
        "description": "Defines how the Fermi energy is calculated, or gives the value",
        "default": "MIDGAP",
    },
    "ENCUT": {
        "description": "The energy cutoff of the plane-wave basis (eV)",
        "default": 500,
    },
    "ALGO": {
        "description": (
            "The electronic minimization algorithm and/or type of GW calculation"
        ),
        "default": "Fast",
    },
    "PREC": {
        "description": "The precision of the FFT grids the real-space projections.",
        "default": "normal",
    },
    "ISMEAR": {
        "description": "How to set the occupancy of each orbital",
        "default": 0,
    },
    "SIGMA": {
        "description": "The width of the electron smearing (eV)",
        "default": 0.1,
    },
    "ISPIN": {
        "description": (
            "Controls whether the calculation is spin-polarized (2) or not (1)"
        ),
        "default": 1,
    },
    "MAGMOM": {
        "description": "The initial magnetic moments on the atoms",
        "default": "1",
    },
    "LNONCOLLINEAR": {
        "description": "Allow noncollinear magnetism",
        "default": ".False.",
    },
    "LASPH": {
        "description": (
            "Include non-spherical contributions from the gradient corrections "
            "inside the PAW spheres"
        ),
        "default": ".True.",
    },
    "LORBIT": {
        "description": (
            "Project the Kohn-Sham orbitals onto local quantum number (lm) to get"
            " on-site charge and magnetic moments"
        ),
        "default": 11,
    },
    "NCORE": {
        "description": "Number of compute cores that work on each orbital",
        "default": 2,
    },
    "KPAR": {
        "description": "Number of compute cores that work on each k-point",
        "default": 1,
    },
    "LPLANE": {
        "description": "Use plane-wise data distribution in real space",
        "default": ".True.",
    },
    "LREAL": {
        "description": (
            "Evaluate the projection operators in real space (Auto) or reciprocal"
            " (.False.)"
        ),
        "default": "Auto",
    },
    "NSIM": {
        "description": (
            "Number of bands to optimize simultaneously when using RMM-DIIS or"
            "the blocked Davidson algorithm (NSIM)"
        ),
        "default": 4,
    },
    "LSCALAPACK": {
        "description": "Use ScaLAPACK for orthonormalization and subspace rotations.",
        "default": ".True.",
    },
    "LSCALU": {
        "description": "Use ScaLAPACK for the subspace rotations.",
        "default": ".False.",
    },
    "ISIF": {
        "description": (
            "Determines whether the stress is calculated, and degrees of freedom."
        ),
        "default": 3,
    },
    "POTIM": {
        "description": "Scaling factor for the step in optimization; timestep for MD",
        "default": 0,
    },
    "NFREE": {
        "description": (
            "Number of remembered steps for RMM-DIIS, or displacements in frozen"
            " phonon calculations."
        ),
        "default": 7,
    },
    "SMASS": {
        "description": (
            "Damping or quenching factor in damped MD; ensemble in normal MD (-3=NVE,"
            " -1=T annealing, >=0 Nosé mass)"
        ),
    },
    "IVDW": {
        "description": (
            "Specifies a vdW dispersion term of the atom-pairwise or many-body type."
        ),
        "default": 12,
    },
    "VDW_S6": {
        "description": (
            "VDW_S6 sets the global scaling factor S6 for the DFT-D2, DFT-D4, DFT-ulg, "
            "and Tkatchenko-Scheffler methods."
        ),
        "default": "",
    },
    "VDW_S8": {
        "description": (
            "VDW_S8 sets the damping function parameter S8 in the DFT-D3 and "
            "DFT-D4 methods."
        ),
        "default": 1.0000,
    },
    "VDW_A1": {
        "description": (
            "VDW_A1 sets the damping function parameter a1 for the DFT-D3 (BJ damping "
            "function) and DFT-D4 methods."
        ),
        "default": "",
    },
    "VDW_A2": {
        "description": (
            "VDW_A2 defines the damping function parameter a2 for the DFT-D3 (BJ "
            "damping function) and DFT-D4 methods."
        ),
        "default": "",
    },
}

"""Properties that VASP produces.
`metadata["results"]` describes the results that this step can produce. It is a
dictionary where the keys are the internal names of the results within this step, and
the values are a dictionary describing the result. For example::

    metadata["results"] = {
        "total_energy": {
            "calculation": [
                "energy",
                "optimization",
            ],
            "description": "The total energy",
            "dimensionality": "scalar",
            "methods": [
                "ccsd",
                "ccsd(t)",
                "dft",
                "hf",
            ],
            "property": "total energy#Psi4#{model}",
            "type": "float",
            "units": "E_h",
        },
    }

Fields
______

calculation : [str]
    Optional metadata describing what subtype of the step produces this result.
    The subtypes are completely arbitrary, but often they are types of calculations
    which is why this is name `calculation`. To use this, the step or a substep
    define `self._calculation` as a value. That value is used to select only the
    results with that value in this field.

description : str
    A human-readable description of the result.

dimensionality : str
    The dimensions of the data. The value can be "scalar" or an array definition
    of the form "[dim1, dim2,...]". Symmetric tringular matrices are denoted
    "triangular[n,n]". The dimensions can be integers, other scalar
    results, or standard parameters such as `n_atoms`. For example, '[3]',
    [3, n_atoms], or "triangular[n_aos, n_aos]".

methods : str
    Optional metadata like the `calculation` data. `methods` provides a second
    level of filtering, often used for the Hamiltionian for *ab initio* calculations
    where some properties may or may not be calculated depending on the type of
    theory.

property : str
    An optional definition of the property for storing this result. Must be one of
    the standard properties defined either in SEAMM or in this steps property
    metadata in `data/properties.csv`.

type : str
    The type of the data: string, integer, or float.

units : str
    Optional units for the result. If present, the value should be in these units.
"""
metadata["results"] = {
    "model": {
        "description": "The model string",
        "dimensionality": "scalar",
        "type": "string",
        "units": "",
        "format": "s",
    },
    "DfE0": {
        "description": "energy of formation",
        "dimensionality": "scalar",
        "property": "DfE0#VASP#{model}",
        "type": "float",
        "units": "kJ/mol",
        "format": ".2f",
    },
    "Ecoh": {
        "description": "cohesive energy",
        "dimensionality": "scalar",
        "property": "Ecoh#VASP#{model}",
        "type": "float",
        "units": "eV",
        "format": ".2f",
    },
    "Ecoh/atom": {
        "description": "cohesive energy per atom",
        "dimensionality": "scalar",
        "type": "float",
        "units": "eV",
        "format": ".2f",
    },
    "energy": {
        "description": "total energy including all terms",
        "dimensionality": "scalar",
        "property": "E#VASP#{model}",
        "type": "float",
        "units": "eV",
        "format": ".3f",
    },
    "energy/atom": {
        "description": "total energy per atom",
        "dimensionality": "scalar",
        "type": "float",
        "units": "eV",
        "format": ".3f",
    },
    "Gelec": {
        "description": "The electronic free energy of the system",
        "dimensionality": "scalar",
        "property": "Gelec#VASP#{model}",
        "type": "float",
        "units": "eV",
        "format": ".3f",
    },
    "gradients": {
        "description": "the Cartesian gradients",
        "dimensionality": ["natoms", 3],
        "property": "gradients#VASP#{model}",
        "type": "float",
        "units": "eV/Å",
        "format": ".3f",
    },
    "P": {
        "description": "pressure",
        "dimensionality": "scalar",
        "property": "pressure#VASP#{model}",
        "type": "float",
        "units": "GPa",
        "format": ".3f",
    },
    "V": {
        "description": "volume",
        "dimensionality": "scalar",
        "property": "volume#VASP#{model}",
        "type": "float",
        "units": "Å^3",
        "format": ".1f",
    },
    "delta V": {
        "description": "change in volume",
        "dimensionality": "scalar",
        "type": "float",
        "units": "Å^3",
        "format": ".1f",
    },
    "density": {
        "description": "density",
        "dimensionality": "scalar",
        "property": "density#VASP#{model}",
        "type": "float",
        "units": "g/mL",
        "format": ".4f",
    },
    "delta density": {
        "description": "change in density",
        "dimensionality": "scalar",
        "type": "float",
        "units": "g/mL",
        "format": ".4f",
    },
    "delta density %": {
        "description": "% change in density",
        "dimensionality": "scalar",
        "type": "float",
        "units": "",
        "format": ".2f",
    },
    "stress": {
        "description": "stress",
        "dimensionality": "[6]",
        "property": "stress#VASP#{model}",
        "type": "float",
        "units": "GPa",
        "format": ".3f",
    },
    "a": {
        "description": "cell parameter 'a'",
        "dimensionality": "scalar",
        "property": "cell_a#VASP#{model}",
        "type": "float",
        "units": "Å",
        "format": ".3f",
    },
    "b": {
        "description": "cell parameter 'b'",
        "dimensionality": "scalar",
        "property": "cell_b#VASP#{model}",
        "type": "float",
        "units": "Å",
        "format": ".3f",
    },
    "c": {
        "description": "cell parameter 'c'",
        "dimensionality": "scalar",
        "property": "cell_c#VASP#{model}",
        "type": "float",
        "units": "Å",
        "format": ".3f",
    },
    "alpha": {
        "description": "cell parameter 'alpha'",
        "dimensionality": "scalar",
        "property": "cell_alpha#VASP#{model}",
        "type": "float",
        "units": "degree",
        "format": ".1f",
    },
    "beta": {
        "description": "cell parameter 'beta'",
        "dimensionality": "scalar",
        "property": "cell_beta#VASP#{model}",
        "type": "float",
        "units": "degree",
        "format": ".1f",
    },
    "gamma": {
        "description": "cell parameter 'gamma'",
        "dimensionality": "scalar",
        "property": "cell_gamma#VASP#{model}",
        "type": "float",
        "units": "degree",
        "format": ".1f",
    },
    "delta a": {
        "description": "change in 'a'",
        "dimensionality": "scalar",
        "type": "float",
        "units": "Å",
    },
    "delta b": {
        "description": "change in 'b'",
        "dimensionality": "scalar",
        "type": "float",
        "units": "Å",
    },
    "delta c": {
        "description": "change in 'c'",
        "dimensionality": "scalar",
        "type": "float",
        "units": "Å",
    },
    "delta alpha": {
        "description": "change in 'alpha'",
        "dimensionality": "scalar",
        "type": "float",
        "units": "degree",
    },
    "delta beta": {
        "description": "change in 'beta'",
        "dimensionality": "scalar",
        "type": "float",
        "units": "degree",
    },
    "delta gamma": {
        "description": "change in 'gamma'",
        "dimensionality": "scalar",
        "type": "float",
        "units": "degree",
    },
    "RMS atom force": {
        "description": "the RMS force on the atoms",
        "dimensionality": "scalar",
        "property": "RMS atom force#VASP#{model}",
        "type": "float",
        "units": "eV/Å",
        "format": ".3f",
    },
    "RMS atom displacement": {
        "calculation": ["optimization"],
        "description": "the RMS displacement of the atoms",
        "dimensionality": "scalar",
        "type": "float",
        "units": "Å",
        "format": ".3f",
    },
    "maximum atom force": {
        "description": "the maximum force on an atom",
        "dimensionality": "scalar",
        "property": "maximum atom force#VASP#{model}",
        "type": "float",
        "units": "eV/Å",
        "format": ".3f",
    },
    "maximum atom force threshold": {
        "calculation": ["optimization"],
        "description": "the maximum force threshold for an atom",
        "dimensionality": "scalar",
        "type": "float",
        "units": "eV/Å",
        "format": ".3f",
    },
    "maximum atom displacement": {
        "calculation": ["optimization"],
        "description": "the maximum displacement of an atom",
        "dimensionality": "scalar",
        "type": "float",
        "units": "Å",
        "format": ".3f",
    },
    "code name": {
        "description": "The name of the code (VASP)",
        "dimensionality": "scalar",
        "type": "string",
        "units": "",
        "format": "",
    },
    "code version": {
        "description": "The version of the VASP code",
        "dimensionality": "scalar",
        "type": "string",
        "units": "",
        "format": "",
    },
    "SEAMM elapsed time": {
        "description": "total elapsed time for VASP",
        "dimensionality": "scalar",
        "type": "float",
        "units": "s",
        "format": ".1f",
    },
    "SEAMM np": {
        "description": "number of processors for VASP",
        "dimensionality": "scalar",
        "type": "integer",
        "format": "d",
    },
}
