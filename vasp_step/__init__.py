# -*- coding: utf-8 -*-

"""
vasp_step
A SEAMM plug-in for VASP
"""

# Bring up the classes so that they appear to be directly in
# the vasp_step package.

from .vasp import VASP  # noqa: F401, E501
from .vasp_step import VASPStep  # noqa: F401, E501
from .tk_vasp import TkVASP  # noqa: F401, E501

from .metadata import metadata  # noqa: F401

from .energy_step import EnergyStep  # noqa: F401
from .energy import Energy  # noqa: F401
from .energy_parameters import EnergyParameters  # noqa: F401
from .tk_energy import TkEnergy  # noqa: F401

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
