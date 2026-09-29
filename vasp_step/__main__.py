# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handle the installation of the VASP step."""

from .installer import Installer


def run():
    """Give the SEAMM installation a vasp.ini describing how to run VASP."""
    installer = Installer()
    installer.run()


if __name__ == "__main__":
    run()
