# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import sys

# e13Tools imports
import e13tools.core as e13c


# %% PYTEST FUNCTIONS
def test_InputError():
    assert Exception in e13c.InputError.mro()
    try:
        raise e13c.InputError
    except Exception:
        assert sys.exc_info()[0] == e13c.InputError


def test_ShapeError():
    assert Exception in e13c.ShapeError.mro()
    try:
        raise e13c.ShapeError
    except Exception:
        assert sys.exc_info()[0] == e13c.ShapeError


def test_compare_version():
    assert e13c.compare_versions('0.1.1', '0.1.0')
    assert not e13c.compare_versions('0.1.0a0', '0.1.0')
    assert not e13c.compare_versions('0.0.9', '0.1.0')
    assert e13c.compare_versions('1.0.0', '0.1.0')
    assert not e13c.compare_versions(None, '0.1.0')
