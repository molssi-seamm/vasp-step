#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `vasp_step` package."""

import pytest  # noqa: F401
import vasp_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = vasp_step.VASP()
    assert str(type(result)) == "<class 'vasp_step.vasp.VASP'>"
