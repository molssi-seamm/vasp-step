# -*- coding: utf-8 -*-

"""Installer for the VASP plug-in.

VASP is licensed software that users install themselves, so this installer does not
install the code. It gives the SEAMM installation a ``vasp.ini`` to edit, from the
template in ``data/vasp.ini``, which says how to run VASP.
"""

import importlib
import logging
import shutil

import seamm_installer

logger = logging.getLogger(__name__)


class Installer(seamm_installer.InstallerBase):
    """Give the installation a vasp.ini describing how to run VASP."""

    def __init__(self, logger=logger):
        super().__init__(logger=logger)

        logger.debug("Initializing the VASP installer object.")

        self.section = "vasp-step"
        self.executables = ["vasp_std"]
        self.resource_path = importlib.resources.files("vasp_step") / "data"

    def exe_version(self, config):
        """Return the name and version of VASP.

        VASP has no option that prints its version, so this only reports whether
        the executable can be found.
        """
        code = config.get("code", "vasp_std")
        found = any(shutil.which(word) for word in code.split())
        return "VASP", "unknown" if found else "not found"
