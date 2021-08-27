# -*- coding: utf-8 -*-

"""
Sampling
========
Provides a collection of functions and techniques useful in sampling problems.
Recommended usage::

    import e13tools.sampling as e13spl

Available submodules
--------------------
:func:`~lhd`
    Provides a Latin Hypercube Sampling method.

"""


# %% IMPORTS
# Module imports
from . import lhs
from .lhs import *

# All declaration
__all__ = []
__all__.extend(lhs.__all__)
