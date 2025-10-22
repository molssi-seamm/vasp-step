# -*- coding: utf-8 -*-
"""
Control parameters for the Energy substep in a VASP step
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class EnergyParameters(seamm.Parameters):  # noqa: E999
    """
    The control parameters for Energy.

    You need to replace the "time" entry in dictionary below these comments with the
    definitions of parameters to control this step. The keys are parameters for the
    current plugin,the values are dictionaries as outlined below.

    Examples
    --------
    ::

        parameters = {
            "time": {
                "default": 100.0,
                "kind": "float",
                "default_units": "ps",
                "enumeration": tuple(),
                "format_string": ".1f",
                "description": "Simulation time:",
                "help_text": ("The time to simulate in the dynamics run.")
            },
        }

    parameters : {str: {str: str}}
        A dictionary containing the parameters for the current step.
        Each key of the dictionary is a dictionary that contains the
        the following keys:

    parameters["default"] :
        The default value of the parameter, used to reset it.

    parameters["kind"] : enum()
        Specifies the kind of a variable. One of  "integer", "float", "string",
        "boolean", or "enum"

        While the "kind" of a variable might be a numeric value, it may still have
        enumerated custom values meaningful to the user. For instance, if the parameter
        is a convergence criterion for an optimizer, custom values like "normal",
        "precise", etc, might be adequate. In addition, any parameter can be set to a
        variable of expression, indicated by having "$" as the first character in the
        field. For example, $OPTIMIZER_CONV.

    parameters["default_units"] : str
        The default units, used for resetting the value.

    parameters["enumeration"]: tuple
        A tuple of enumerated values.

    parameters["format_string"]: str
        A format string for "pretty" output.

    parameters["description"]: str
        A short string used as a prompt in the GUI.

    parameters["help_text"]: str
        A longer string to display as help for the user.

    See Also
    --------
    Energy, TkEnergy, Energy EnergyParameters, EnergyStep
    """

    parameters = {
        "input only": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": (
                "yes",
                "no",
            ),
            "format_string": "s",
            "description": "Write the input files and stop:",
            "help_text": "Don't run VASP. Just write the input files.",
        },
        "calculate stress": {
            "default": "yes",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Calculate stress:",
            "help_text": (
                "Whether to calculate the stress in a single-point calculation."
            ),
        },
        # Convergence and precision parameters
        "nelm": {
            "default": 60,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "Maximum electronic iterations:",
            "help_text": (
                "Maximum number of steps for the selfconsistent electronic"
                " minimization."
            ),
        },
        "ediff": {
            "default": 1.0e-6,
            "kind": "float",
            "default_units": "",
            "enumeration": None,
            "format_string": ".2E",
            "description": "Electronic convergence threshold",
            "help_text": "Threshold for the electronic convergence.",
        },
        "electronic method": {
            "default": "fast",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": (
                "normal" "fast",
                "very fast",
                "conjugate",
                "damped",
                "exact",
            ),
            "format_string": "",
            "description": "Electronic minimization method:",
            "help_text": "The algorithm used tp minimize the electronic energy.",
        },
        # Performance
        "ncore": {
            "default": 2,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "Cores per orbital:",
            "help_text": "Number of cores working on each orbital (NCORE).",
        },
        "kpar": {
            "default": 2,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "Cores per k-point:",
            "help_text": "Number of cores working on each k-point (KPAR).",
        },
        "lplane": {
            "default": "yes",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Plane-wise data distribution:",
            "help_text": (
                "Whether to use the plane-wise data distribution, which reduces"
                " memory bandwidth but worsens load balancing."
            ),
        },
        "lreal": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Use real space projection:",
            "help_text": (
                "Whether to use projection operators in real space (True) or"
                " reciprocal (False)."
            ),
        },
        # Miscellaneous
        "efermi": {
            "default": "middle of the gap",
            "kind": "float",
            "default_units": "eV",
            "enumeration": ("middle of the gap", "legacy"),
            "format_string": ".3f",
            "description": "Fermi level:",
            "help_text": (
                "The position of the Fermi level: either the middle of the gap,"
                " the previous legacy value, or a numerical value in eV"
            ),
        },
        "lorbit": {
            "default": "yes",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Calculate atomic densities and spins:",
            "help_text": (
                "Whether to calculate the angular momemntum resolved densities and"
                " spins on the atoms."
            ),
        },
        # Keywords and Results handling
        "extra keywords": {
            "default": [],
            "kind": "list",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Extra keywords",
            "help_text": ("Extra keywords to add/overwrite those from the GUI. "),
        },
        "results": {
            "default": {},
            "kind": "dictionary",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "results",
            "help_text": "The results to save to variables or in tables.",
        },
    }

    model_parameters = {
        "model": {
            "default": "Generalized-gradient approximations (GGA)",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": (
                "Local-density approximation (LDA)",
                "Generalized-gradient approximations (GGA)",
                "Meta-generalized gradient approximations (meta-GGA)",
                "Hartree-Fock and hybrid functionals",
                "Hybrid meta generalized gradient approximation (Hybrid meta-GGA)",
            ),
            "format_string": "",
            "description": "Computational model:",
            "help_text": "The high-level computational model employed.",
        },
        "submodel": {
            "default": "PBE : Perdew, Burke and Ernzerhof",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Model version:",
            "help_text": "The version of the model employed.",
        },
        "set of potentials": {
            "default": "potpaw_PBE.64",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "Set of potentials:",
            "help_text": "The set of PAW potentials to use.",
        },
        "potentials": {
            "default": {},
            "kind": "dictionary",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "PAW potentials",
            "help_text": "The PAW potentials to use for all systems",
        },
        "subset": {
            "default": "Show all potentials",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": (
                "Show all potentials",
                "Show standard potentials",
                "Show GW potentials",
            ),
            "format_string": "",
            "description": "Subset of potentials:",
            "help_text": "The subset of PAW potentials to use.",
        },
        "plane-wave cutoff": {
            "default": 500,
            "kind": "float",
            "default_units": "eV",
            "enumeration": None,
            "format_string": "",
            "description": "Plane-wave cutoff:",
            "help_text": "The energy cutoff for the plane-wave basis.",
        },
        "enmax": {
            "default": 0,
            "kind": "float",
            "default_units": "eV",
            "enumeration": None,
            "format_string": "",
            "description": "Maximum cutoff in potentials:",
            "help_text": "The maximum energy cutoff in the specified potentials.",
        },
        "spin polarization": {
            "default": "collinear",
            "kind": "string",
            "default_units": "",
            "enumeration": ("none", "collinear", "noncollinear"),
            "format_string": "",
            "description": "Spin polarization:",
            "help_text": "Whether and what type of spin-polarization to use",
        },
        "magnetic moments": {
            "default": "default",
            "kind": "string",
            "default_units": "",
            "enumeration": ("default"),
            "format_string": "",
            "description": "Magnetic moments:",
            "help_text": "The initial magnetic moments on the atoms",
        },
        "nonspherical PAW": {
            "default": "yes",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Include non-spherical terms in PAW:",
            "help_text": (
                "Include non-spherical contributions from the gradient corrections "
                "inside the PAW spheres"
            ),
        },
    }

    kspace_parameters = {
        "k-grid method": {
            "default": "grid spacing",
            "kind": "string",
            "default_units": "",
            "enumeration": ("grid spacing", "explicit grid dimensions"),
            "format_string": "",
            "description": "Specify k-space grid using:",
            "help_text": "How to specify the k-space integration grid.",
        },
        "k-spacing": {
            "default": 0.5,
            "kind": "float",
            "default_units": "1/Å",
            "enumeration": None,
            "format_string": "",
            "description": "K-spacing:",
            "help_text": "The spacing of the grid in reciprocal space.",
        },
        "odd grid": {
            "default": "yes",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Force to odd numbers:",
            "help_text": "Whether to force the grid sizes to odd numbers.",
        },
        "centering": {
            "default": "Monkhorst-Pack",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": (
                "𝚪-centered",
                "Monkhorst-Pack",
            ),
            "format_string": "",
            "description": "Centering of grid:",
            "help_text": "How to center the grid in reciprocal space.",
        },
        "occupation type": {
            "default": "Methfessel-Paxton method",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": (
                "Gaussian smearing",
                "the Methfessel-Paxton method",
                "Fermi-Dirac smearing",
                "the tetrahedron method without smearing",
                "the tetrahedron method with Blöchl corrections without smearing",
                "the tetrahedron method with Fermi-Dirac smearing",
                (
                    "the tetrahedron method with Blöchl corrections with Fermi-Dirac"
                    " smearing"
                ),
            ),
            "format_string": "",
            "description": "Smearing:",
            "help_text": (
                "How occupy the orbitals, typically smearing the electrons as they "
                "would be at finite temperature."
            ),
        },
        "Methfessel-Paxton order": {
            "default": 2,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "Order:",
            "help_text": (
                "The order of the expansion in the Methfessel-Paxton method."
            ),
        },
        "smearing width": {
            "default": 0.01,
            "kind": "float",
            "default_units": "eV",
            "enumeration": None,
            "format_string": ".3f",
            "description": "Width:",
            "help_text": "The smearing or broadening width.",
        },
        "na": {
            "default": 4,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "NPoints in a:",
            "help_text": (
                "Number of points in the first direction of the Brillouin zone."
            ),
        },
        "nb": {
            "default": 4,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "b:",
            "help_text": (
                "Number of points in the second direction of the Brillouin zone."
            ),
        },
        "nc": {
            "default": 4,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "c:",
            "help_text": (
                "Number of points in the third direction of the Brillouin zone."
            ),
        },
    }

    def __init__(self, defaults={}, data=None):
        """
        Initialize the parameters, by default with the parameters defined above

        Parameters
        ----------
        defaults: dict
            A dictionary of parameters to initialize. The parameters
            above are used first and any given will override/add to them.
        data: dict
            A dictionary of keys and a subdictionary with value and units
            for updating the current, default values.

        Returns
        -------
        None
        """

        logger.debug("EnergyParameters.__init__")

        super().__init__(
            defaults={
                **EnergyParameters.parameters,
                **EnergyParameters.model_parameters,
                **EnergyParameters.kspace_parameters,
                **defaults,
            },
            data=data,
        )
