# -*- coding: utf-8 -*-

"""Resolve VASP on the machine that runs it (seamm_exec's resolver hook).

A VASP task's command names ``{gamma_code}`` (and ``{dftd4}`` for a -D4
functional). Where the task runs, seamm_exec reads that machine's
``<root>/vasp.ini`` ``[local]`` section and calls :func:`resolve`, which puts the
configured commands into the command; they may themselves use ``{NTASKS}``,
which the executor then fills in (e.g. ``gamma_code = mpiexec -np {NTASKS}
vasp_gam``).

``dftd4`` is the dftd4 program (conda-forge ``dftd4``, version 4; on
TinkerCliffs ``/projects/seamm/conda-envs/dftd4/bin/dftd4``).

Registered as the entry point ``vasp`` in ``org.molssi.seamm.exec.resolvers``.
"""

import shutil


def resolve(config, cmd, env, ce, root):
    """``(config, cmd, env)`` for running VASP here.

    Parameters
    ----------
    config : dict
        This machine's ``vasp.ini`` section (empty if there is none).
    cmd : [str]
        The task's command template.
    env : dict
        The task's extra environment.
    ce : dict
        The task's computational environment (``NTASKS``, ...).
    root : str or Path
        The SEAMM root holding ``vasp.ini``.
    """
    config = dict(config)
    gamma = (config.get("gamma_code") or "").strip()
    code = (config.get("code") or "").strip()
    if gamma == "":
        gamma = code
    if gamma == "":
        if shutil.which("vasp_gam") is None:
            raise RuntimeError(
                "Could not find VASP: set 'gamma_code' (e.g. 'mpiexec -np {NTASKS} "
                f"vasp_gam') in the [local] section of {root}/vasp.ini."
            )
        gamma = "mpiexec -np {NTASKS} vasp_gam"

    resolved = []
    for word in cmd:
        if word == "{gamma_code}":
            resolved.append(gamma)
        elif word == "{dftd4}":
            dftd4 = (config.get("dftd4") or "").strip() or shutil.which("dftd4")
            if not dftd4:
                raise RuntimeError(
                    "A -D4 functional needs the dftd4 program: set 'dftd4' (its "
                    f"full path) in the [local] section of {root}/vasp.ini, or put "
                    "it on the PATH."
                )
            resolved.append(dftd4)
        else:
            resolved.append(word)
    return config, resolved, dict(env)


def _available(root):
    return shutil.which("vasp_gam") is not None or shutil.which("vasp_std") is not None


resolve.available = _available
