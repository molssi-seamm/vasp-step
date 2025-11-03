# -*- coding: utf-8 -*-
"""
Control parameters for the Optimization step in a SEAMM flowchart
"""

import logging

from .energy_parameters import EnergyParameters

logger = logging.getLogger(__name__)


class OptimizationParameters(EnergyParameters):  # noqa: E999
    """
    The control parameters for Optimization.

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
    Optimization, TkOptimization, Optimization OptimizationParameters, OptimizationStep
    """

    parameters = {
        "optimization method": {
            "default": "RMM-DIIS",
            "kind": "string",
            "default_units": "",
            "enumeration": ("RMM-DIIS", "Conjugate Gradients", "Damped MD"),
            "format_string": "",
            "description": "Optimization algorithm:",
            "help_text": "The optimization algorithm to employ (IBRION).",
        },
        "optimize atom positions": {
            "default": "yes",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Optimize the atomic positions:",
            "help_text": (
                "Whether to allow the atom positions (in fractional coordinates) to"
                "change during the optimization."
            ),
        },
        "optimize cell shape": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Optimize the call shape:",
            "help_text": (
                "Whether to allow the shape of the cell to"
                "change during the optimization."
            ),
        },
        "optimize cell volume": {
            "default": "no",
            "kind": "boolean",
            "default_units": "",
            "enumeration": ("yes", "no"),
            "format_string": "",
            "description": "Optimize the call volume:",
            "help_text": (
                "Whether to allow the volume of the cell to"
                "change during the optimization."
            ),
        },
        "number of steps": {
            "default": 100,
            "kind": "integer",
            "default_units": "",
            "enumeration": None,
            "format_string": "",
            "description": "Maximum number of steps:",
            "help_text": "The maximum number of optimization steps (NSW).",
        },
        "convergence cutoff": {
            "default": 1.0e-02,
            "kind": "float",
            "default_units": "eV/Å",
            "enumeration": None,
            "format_string": "",
            "description": "Gradient convergence:",
            "help_text": (
                "The convergence in terms of RMS of the gradient (sqrt(EDIFFG)."
            ),
        },
        "step scale factor": {
            "default": 0.5,
            "kind": "float",
            "default_units": "",
            "enumeration": None,
            "format_string": ".1f",
            "description": "Factor to scale optimization steps:",
            "help_text": (
                "The step length predicted by the optimizer is scaled by this factor"
                " (POTIM)."
            ),
        },
        "iteration history length": {
            "default": "default",
            "kind": "integer",
            "default_units": "",
            "enumeration": ("default",),
            "format_string": "",
            "description": "Length of iteration history used:",
            "help_text": (
                "How many iterations are remembered when using RMM-DIIS (NFREE)."
            ),
        },
        "damped MD approach": {
            "default": "damping",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": ("damping", "quenching"),
            "format_string": "",
            "description": "How to damp the velocity:",
            "help_text": "How the velocity is damped during damped MD",
        },
        "force scale factor": {
            "default": 1,
            "kind": "float",
            "default_units": "",
            "enumeration": None,
            "format_string": ".1f",
            "description": "Force scale factor:",
            "help_text": "The factor to scale the force in damped MD (SMASS).",
        },
        "velocity scale factor": {
            "default": 0.8,
            "kind": "float",
            "default_units": "",
            "enumeration": None,
            "format_string": ".1f",
            "description": "Damping factor:",
            "help_text": "The factor for scaling the previous velocity (0->1)",
        },
        "velocity quenching factor": {
            "default": 1.0,
            "kind": "float",
            "default_units": "",
            "enumeration": None,
            "format_string": ".1f",
            "description": "Quenching factor:",
            "help_text": "The factor to quench the velocity in damped MD (POTIM).",
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

        logger.debug("OptimizationParameters.__init__")

        super().__init__(
            defaults={**OptimizationParameters.parameters, **defaults}, data=data
        )
