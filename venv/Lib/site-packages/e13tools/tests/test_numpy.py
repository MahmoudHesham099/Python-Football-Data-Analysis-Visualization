# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import numpy as np
import pytest

# e13Tools imports
from e13tools import InputError, ShapeError
from e13tools.numpy import (
    diff, intersect, isin, rot90, setdiff, setxor, sort2D, transposeC, union)


# %% PYTEST HELPERS
@pytest.fixture(scope='module')
def array1():
    return(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))


@pytest.fixture(scope='module')
def array2():
    return(np.array([[[1, 2], [3, 4]], [[5, 6], [3, 4]]]))


# %% PYTEST FUNCTIONS
# Pytest class for the diff()-function
class Test_diff(object):
    # Test row difference between two matrices
    def test_matrices_row(self):
        mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        assert np.allclose(diff(mat1, mat2),
                           np.array([[[-3, -3, -3], [-6, -6, -6]],
                                     [[0, 0, 0], [-3, -3, -3]]]))

    # Test column difference between two matrices
    def test_matrices_column(self):
        mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        assert np.allclose(diff(mat1, mat2, axis=1),
                           np.array([[[-3, -3], [-4, -4], [-5, -5]],
                                     [[-2, -2], [-3, -3], [-4, -4]],
                                     [[-1, -1], [-2, -2], [-3, -3]]]))

    # Test difference of matrix with itself
    def test_single_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(diff(mat), np.array([[-3, -3, -3]]))
        assert np.allclose(diff(mat, flatten=True), np.array([[-3, -3, -3]]))
        assert np.allclose(diff(mat, flatten=False),
                           np.array([[[0, 0, 0], [-3, -3, -3]],
                                     [[3, 3, 3], [0, 0, 0]]]))

    # Test difference between matrix and vector
    def test_matrix_vector(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([7, 8, 9])
        assert np.allclose(diff(mat, vec), np.array([[-6, -6, -6],
                                                     [-3, -3, -3]]))
        assert np.allclose(diff(vec, mat), np.array([[6, 6, 6],
                                                     [3, 3, 3]]))

    # Test difference between two vectors
    def test_vectors(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        assert np.allclose(diff(vec1, vec2), np.array([[-3, -4, -5],
                                                       [-2, -3, -4],
                                                       [-1, -2, -3]]))

    # Test difference of vector with itself
    def test_single_vector(self):
        vec = np.array([7, 8, 9])
        assert np.allclose(diff(vec), [-1, -2, -1])

    # Test difference bwteen two scalars
    def test_scalars(self):
        assert diff(2, 1) == 1

    # Test difference of scalar with itself
    def test_single_scalar(self):
        assert diff(1) == 0

    # Test if invalid axis raises an error using a single vector
    def test_single_invalid_axis(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(InputError):
            diff(vec, axis=1)

    # Test if invalid axis raises an error using two vectors
    def test_double_invalid_axis(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        with pytest.raises(InputError):
            diff(vec1, vec2, axis=1)

    # Test if using matrices with different axes lengths raises an error
    def test_two_diff_axes(self):
        mat1 = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        with pytest.raises(ShapeError):
            diff(mat1, mat2)

    # Test if using matrix and vector with invalid axis raises an error
    def test_matrix_vector_invalid_axis(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([7, 8, 9])
        with pytest.raises(InputError):
            diff(mat, vec, axis=2)

    # Test if using matrix and vector with different axes lengths raises error
    def test_matrix_vector_diff_axes(self):
        mat = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            diff(mat, vec)


# Pytest class for intersect()-function
class Test_intersect(object):
    # Test if default works
    def test_default(self, array1, array2):
        assert np.allclose(intersect(array1, array2),
                           np.array([[[1, 2], [3, 4]]]))

    # Test if a flattened array is returned for axis=None
    def test_axis_None(self, array1, array2):
        assert np.allclose(intersect(array1, array2, None),
                           np.array([1, 2, 3, 4, 5, 6]))

    # Test for axis=1
    def test_axis_1(self, array1, array2):
        assert np.allclose(intersect(array1, array2, 1),
                           np.array([[[1, 2]], [[5, 6]]]))

    # Test for axis=2
    def test_axis_2(self, array1, array2):
        assert np.allclose(intersect(array1, array2, 2),
                           np.array([]))


# Pytest class for isin()-function
class Test_isin(object):
    # Test if default works
    def test_default(self, array1, array2):
        assert np.allclose(isin(array1, array2),
                           np.array([True, False]))

    # Test if a non-flattened array is returned for axis=None
    def test_axis_None(self, array1, array2):
        assert np.allclose(isin(array1, array2, None),
                           np.array([[[True, True], [True, True]],
                                     [[True, True], [False, False]]]))

    # Test for axis=1
    def test_axis_1(self, array1, array2):
        assert np.allclose(isin(array1, array2, 1),
                           np.array([True, False]))

    # Test for axis=2
    def test_axis_2(self, array1, array2):
        assert np.allclose(isin(array1, array2, 2),
                           np.array([False, False]))

    # Test for invalid axis
    def test_invalid_axis(self, array1, array2):
        with pytest.raises(InputError):
            isin(array1, array2, axis=3)


# Pytest class for rot90()-function
class Test_rot90(object):
    # Test if rotating with default values returns correct array
    def test_default(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array), np.array([[1, 0.75],
                                                   [0, 0.25],
                                                   [0.25, 1],
                                                   [0.75, 0]]))

    # Test if rotating 0 times with default values returns correct array
    def test_default_0(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=0), np.array([[0.75, 0],
                                                            [0.25, 1],
                                                            [1, 0.75],
                                                            [0, 0.25]]))

    # Test if rotating 1 time with default values returns correct array
    def test_default_1(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=1), np.array([[1, 0.75],
                                                            [0, 0.25],
                                                            [0.25, 1],
                                                            [0.75, 0]]))

    # Test if rotating 2 times with default values returns correct array
    def test_default_2(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=2), np.array([[0.25, 1],
                                                            [0.75, 0],
                                                            [0, 0.25],
                                                            [1, 0.75]]))

    # Test if rotating 3 times with default values returns correct array
    def test_default_3(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, n_rot=3), np.array([[0, 0.25],
                                                            [1, 0.75],
                                                            [0.75, 0],
                                                            [0.25, 1]]))

    # Test if changing the 2D rotation axis returns correct array
    def test_2D_rot_axis(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        assert np.allclose(rot90(array, rot_axis=[0.2, 0.7]),
                           np.array([[0.9, 1.25], [-0.1, 0.75],
                                     [0.15, 1.5], [0.65, 0.5]]))

    # Test if changing the 3D rotation axis returns correct array
    def test_3D_rot_axis(self):
        array = np.array([[0.75, 0, 0], [0.25, 1, 0], [1, 0.75, 0]])
        assert np.allclose(rot90(array, rot_axis=[0.2, 0.7, 0]),
                           np.array([[0.9, 1.25, 0], [-0.1, 0.75, 0],
                                     [0.15, 1.5, 0]]))

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            rot90(vec)

    # Test if using three axes for rotation raises an error
    def test_3_rot_axes(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        with pytest.raises(InputError):
            rot90(array, axes=(0, 1, 2))

    # Test if using an invalid rotation axis string raises an error
    def test_invalid_rot_axis_str(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        with pytest.raises(ValueError):
            rot90(array, rot_axis='test')

    # Test if changing the 3D rotation axis incorrectly raises an error
    def test_invalid_3D_rot_axis(self):
        array = np.array([[0.75, 0, 0], [0.25, 1, 0], [1, 0.75, 0]])
        with pytest.raises(ValueError):
            rot90(array, rot_axis=[0.2, 0, 0])

    # Test if changing using incorrect rotation axis raises an error
    def test_4D_rot_axis(self):
        array = np.array([[0.75, 0, 0], [0.25, 1, 0], [1, 0.75, 0]])
        with pytest.raises(ShapeError):
            rot90(array, rot_axis=[0.2, 0.7, 0, 0])

    # Test if rotating 4.5 times with default values raises an error
    def test_default_4_5(self):
        array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        with pytest.raises(InputError):
            rot90(array, n_rot=4.5)


# Pytest class for setdiff()-function
class Test_setdiff(object):
    # Test if default works
    def test_default(self, array1, array2):
        assert np.allclose(setdiff(array1, array2),
                           np.array([[[5, 6], [7, 8]]]))

    # Test if a flattened array is returned for axis=None
    def test_axis_None(self, array1, array2):
        assert np.allclose(setdiff(array1, array2, None),
                           np.array([7, 8]))

    # Test for axis=1
    def test_axis_1(self, array1, array2):
        assert np.allclose(setdiff(array1, array2, 1),
                           np.array([[[3, 4]], [[7, 8]]]))

    # Test for axis=2
    def test_axis_2(self, array1, array2):
        assert np.allclose(setdiff(array1, array2, 2),
                           array1)


# Pytest class for setxor()-function
class Test_setxor(object):
    # Test if default works
    def test_default(self, array1, array2):
        assert np.allclose(setxor(array1, array2),
                           np.array([[[5, 6], [3, 4]], [[5, 6], [7, 8]]]))

    # Test if a flattened array is returned for axis=None
    def test_axis_None(self, array1, array2):
        assert np.allclose(setxor(array1, array2, None),
                           np.array([7, 8]))

    # Test for axis=1
    def test_axis_1(self, array1, array2):
        assert np.allclose(setxor(array1, array2, 1),
                           np.array([[[3, 4], [3, 4]], [[3, 4], [7, 8]]]))

    # Test for axis=2
    def test_axis_2(self, array1, array2):
        assert np.allclose(setxor(array1, array2, 2),
                           np.array([[[1, 1, 2, 2], [3, 3, 4, 4]],
                                     [[5, 5, 6, 6], [3, 7, 4, 8]]]))


# Pytest class for union()-function
class Test_union(object):
    # Test if default works
    def test_default(self, array1, array2):
        assert np.allclose(union(array1, array2),
                           np.array([[[1, 2], [3, 4]],
                                     [[5, 6], [3, 4]],
                                     [[5, 6], [7, 8]]]))

    # Test if a flattened array is returned for axis=None
    def test_axis_None(self, array1, array2):
        assert np.allclose(union(array1, array2, None),
                           np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    # Test for axis=1
    def test_axis_1(self, array1, array2):
        assert np.allclose(union(array1, array2, 1),
                           np.array([[[1, 2], [3, 4], [3, 4]],
                                     [[5, 6], [3, 4], [7, 8]]]))

    # Test for axis=2
    def test_axis_2(self, array1, array2):
        assert np.allclose(union(array1, array2, 2),
                           np.array([[[1, 1, 2, 2], [3, 3, 4, 4]],
                                     [[5, 5, 6, 6], [3, 7, 4, 8]]]))


# Pytest class for sort2D()-function
class Test_sort2D(object):
    # Test with default values
    def test_default(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array), np.array([[0, 1, 1], [0, 4, 6],
                                                    [3, 5, 8], [7, 13, 9]]))

    # Test if sorting on first column works
    def test_first_col(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=0),
                           np.array([[0, 5, 1], [0, 1, 8],
                                     [3, 13, 6], [7, 4, 9]]))

    # Test if sorting on first row works
    def test_first_row(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, axis=0, order=0),
                           np.array([[0, 1, 5], [7, 9, 4],
                                     [3, 6, 13], [0, 8, 1]]))

    # Test if sorting in order works
    def test_in_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=(0, 1, 2)),
                           np.array([[0, 1, 8], [0, 5, 1],
                                     [3, 13, 6], [7, 4, 9]]))

    # Test if sorting in different order works
    def test_diff_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        assert np.allclose(sort2D(array, order=(0, 2, 1)),
                           np.array([[0, 5, 1], [0, 1, 8],
                                     [3, 13, 6], [7, 4, 9]]))

    # Test if using vector raises an error
    def test_vector(self):
        vec = np.array([7, 8, 9])
        with pytest.raises(ShapeError):
            sort2D(vec)

    # Test if using invalid axis raises an error
    def test_invalid_axis(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        with pytest.raises(InputError):
            sort2D(array, axis=3)

    # Test if sorting in invalid order raises an error
    def test_invalid_order(self):
        array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        with pytest.raises(ValueError):
            sort2D(array, order=(0, 3, 1))


# Pytest class for transposeC()-function
class Test_transposeC(object):
    # Test if transposing a real array returns the correct transpose
    def test_real(self):
        array = np.array([[1, 2.5], [3.5, 5]])
        assert np.allclose(transposeC(array), np.array([[1, 3.5], [2.5, 5]]))

    # Test if transposing a complex array returns the correct transpose
    def test_complex(self):
        array = np.array([[1, -2+4j], [7.5j, 0]])
        assert np.allclose(transposeC(array), np.array([[1-0j, 0-7.5j],
                                                        [-2-4j, 0-0j]]))
