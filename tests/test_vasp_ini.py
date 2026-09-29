#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""vasp.ini: the shipped template and how the step reads it."""

import configparser
import importlib.resources

import pytest

import vasp_step
from vasp_step import energy

TEMPLATE = importlib.resources.files("vasp_step") / "data" / "vasp.ini"


def _step():
    """An Energy node without a flowchart; _vasp_config needs no state."""
    return vasp_step.Energy.__new__(vasp_step.Energy)


def test_template_is_shipped_and_leaves_the_commands_to_the_user():
    config = configparser.ConfigParser()
    config.read_string(TEMPLATE.read_text())
    assert config["local"]["installation"] == "local"
    assert config["local"].get("code", "") == ""
    assert "{NTASKS}" in TEMPLATE.read_text()


def test_missing_file_gets_the_template_and_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(energy.shutil, "which", lambda name: None)
    path = tmp_path / "vasp.ini"
    with pytest.raises(RuntimeError, match="SEAMM does not know how to run VASP"):
        _step()._vasp_config("local", path)
    text = path.read_text()
    assert "noncollinear_code" in text and "srun" in text  # the comments survive


def test_commands_found_on_the_path_are_saved(tmp_path, monkeypatch):
    found = {"vasp_std": "/opt/vasp/vasp_std", "vasp_gam": "/opt/vasp/vasp_gam"}
    monkeypatch.setattr(energy.shutil, "which", lambda name: found.get(name))
    path = tmp_path / "vasp.ini"
    config = _step()._vasp_config("local", path)
    assert config["code"] == "mpiexec -np {NTASKS} vasp_std"
    assert config["gamma_code"] == "mpiexec -np {NTASKS} vasp_gam"
    assert "noncollinear_code" not in config  # no vasp_ncl on the PATH
    saved = configparser.ConfigParser()
    saved.read(path)
    assert saved["local"]["code"] == "mpiexec -np {NTASKS} vasp_std"


def test_given_commands_are_used_as_they_are(tmp_path, monkeypatch):
    monkeypatch.setattr(energy.shutil, "which", lambda name: pytest.fail("no search"))
    path = tmp_path / "vasp.ini"
    mine = (
        "[local]\ninstallation = modules\nmodules = VASP/6.5.1\n"
        "code = srun -n {NTASKS} vasp_std\n"
    )
    path.write_text(mine)
    config = _step()._vasp_config("local", path)
    assert config["code"] == "srun -n {NTASKS} vasp_std"
    assert path.read_text() == mine


def test_installer_points_at_the_template():
    pytest.importorskip("seamm_installer")  # from the SEAMM Manager
    from vasp_step.installer import Installer

    assert Installer.__init__  # imports with the SEAMM installer base
    assert (importlib.resources.files("vasp_step") / "data" / "vasp.ini").is_file()
    assert (
        importlib.resources.files("vasp_step") / "data" / "configuration.txt"
    ).is_file()
