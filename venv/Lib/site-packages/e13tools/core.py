# -*- coding: utf-8 -*-

"""
Core
====
Provides a collection of functions that are core to **e13Tools** and are
imported automatically.

"""


# %% IMPORTS
# Built-in imports
from pkg_resources import parse_version

# All declaration
__all__ = ['InputError', 'ShapeError', 'compare_versions']


# %% CLASSES
# Define Error class for wrong inputs
class InputError(Exception):
    """
    Generic exception raised for errors in the function input arguments.

    General purpose exception class, raised whenever the function input
    arguments prevent the correct execution of the function without specifying
    the type of error (eg. ValueError, TypeError, etc).

    """

    pass


# Define Error class for wrong shapes
class ShapeError(Exception):
    """
    Inappropriate argument shape (of correct type).

    """

    pass


# %% FUNCTIONS
# Function that compares two versions with each other
def compare_versions(a, b):
    """
    Compares provided versions `a` and `b` with each other, and returns *True*
    if version `a` is later than or equal to version `b`.

    """

    if a:
        return(parse_version(a) >= parse_version(b))
    else:
        return(False)
