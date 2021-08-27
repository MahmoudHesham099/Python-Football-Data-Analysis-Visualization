# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import numpy as np
import pytest

# e13Tools imports
from e13tools import ShapeError
from e13tools.math import gcd, is_PD, lcm, nCr, nearest_PD, nPr


# %% PYTEST FUNCTIONS
# Do default test for gcd()-function
def test_gcd():
    assert gcd(18, 60, 72, 138) == 6


# Pytest class for the is_PD()-function
class Test_is_PD(object):
    # Test if real PD matrix returns True
    def test_real_PD_matrix(self):
        mat = np.eye(3)
        assert is_PD(mat)

    # Test if real non-PD matrix returns False
    def test_real_non_PD_matrix(self):
        mat = np.array([[1, 2.5], [2.5, 4]])
        assert not is_PD(mat)

    # Test if complex PD matrix returns True
    def test_complex_PD_matrix(self):
        mat = np.array([[4, 1.5+1j], [1.5-1j, 3]])
        assert is_PD(mat)

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            is_PD(vec)

    # Test if using a non-square matrix raises an error
    def test_non_square_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ShapeError):
            is_PD(mat)

    # Test if non-Hermitian matrix raises an error
    def test_non_Hermitian_matrix(self):
        mat = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            is_PD(mat)


# Do default test for lcm()-function
def test_lcm():
    assert lcm(8, 9, 21) == 504


# Pytest class for nCr()-function
class Test_nCr(object):
    # Test for repeat = False
    def test_no_repeat(self):
        assert nCr(4, 0) == 1
        assert nCr(4, 1) == 4
        assert nCr(4, 2) == 6
        assert nCr(4, 3) == 4
        assert nCr(4, 4) == 1
        assert nCr(4, 5) == 0

    # Test for repeat = True
    def test_with_repeat(self):
        assert nCr(4, 0, repeat=True) == 1
        assert nCr(4, 1, repeat=True) == 4
        assert nCr(4, 2, repeat=True) == 10
        assert nCr(4, 3, repeat=True) == 20
        assert nCr(4, 4, repeat=True) == 35
        assert nCr(4, 5, repeat=True) == 56


# Pytest class for nearest_PD()-function
class Test_nearest_PD(object):
    # Test if using a real PD matrix returns the matrix
    def test_real_PD_matrix(self):
        mat = np.eye(3)
        assert is_PD(mat)
        assert np.allclose(nearest_PD(mat), mat)

    # Test if using a real non-PD matrix converts it into a PD matrix
    def test_real_non_PD_matrix(self):
        mat = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            is_PD(mat)
        mat_PD = nearest_PD(mat)
        assert is_PD(mat_PD)
        assert np.allclose(mat_PD, np.array([[1.31461828, 2.32186616],
                                             [2.32186616, 4.10085767]]))

    # Test if using a complex non-PD matrix converts it into a PD matrix
    def test_complex_non_PD_matrix(self):
        mat = np.array([[4, 2+1j], [1+3j, 3]])
        mat_PD = nearest_PD(mat)
        assert is_PD(mat_PD)
        assert np.allclose(mat_PD, np.array([[4.0+0.j, 1.5-1.j],
                                             [1.5+1.j, 3.0+0.j]]))

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            nearest_PD(vec)

    # Test if using a non-square matrix raises an error
    def test_non_square_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ShapeError):
            nearest_PD(mat)


# Pytest class for nPr()-function
class Test_nPr(object):
    # Test for repeat = False
    def test_no_repeat(self):
        assert nPr(4, 0) == 1
        assert nPr(4, 1) == 4
        assert nPr(4, 2) == 12
        assert nPr(4, 3) == 24
        assert nPr(4, 4) == 24
        assert nPr(4, 5) == 0

    # Test for repeat = True
    def test_with_repeat(self):
        assert nPr(4, 0, repeat=True) == 1
        assert nPr(4, 1, repeat=True) == 4
        assert nPr(4, 2, repeat=True) == 16
        assert nPr(4, 3, repeat=True) == 64
        assert nPr(4, 4, repeat=True) == 256
        assert nPr(4, 5, repeat=True) == 1024
