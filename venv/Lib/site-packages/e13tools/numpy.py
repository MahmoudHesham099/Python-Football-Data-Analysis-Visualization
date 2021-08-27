# -*- coding: utf-8 -*-

"""
NumPy
=====
Provides a collection of functions useful in manipulating *NumPy* arrays.

"""


# %% IMPORTS
# Package imports
import numpy as np

# e13Tools imports
from e13tools.core import InputError, ShapeError

# All declaration
__all__ = ['diff', 'intersect', 'isin', 'rot90', 'setdiff', 'setxor', 'sort2D',
           'transposeC', 'union']


# %% FUNCTIONS
# This function calculates the pair-wise differences between two inputs
def diff(array1, array2=None, axis=0, flatten=True):
    """
    Calculates the pair-wise differences between inputs `array1` and `array2`
    over the given `axis`.

    Parameters
    ----------
    array1 : array_like
        One of the inputs used to calculate the pair-wise differences.

    Optional
    --------
    array2 : array_like or None. Default: None
        The other input used to calculate the pair-wise differences.
        If *None*, `array2` is equal to `array1`.
        If not *None*, the length of all axes except `axis` must be equal for
        both arrays.
    axis : int. Default: 0
        Over which axis to calculate the pair-wise differences. Default is over
        the first axis. A negative value counts from the last to the first
        axis.
    flatten : bool. Default: True
        If `array2` is *None*, whether or not to calculate all pair-wise
        differences.
        If *True*, a flattened array containing all above-diagonal pair-wise
        differences is returned. This is useful if only off-diagonal terms are
        required and the sign is not important.
        If *False*, an array with all pair-wise differences is returned.

    Returns
    -------
    diff_array : :obj:`~numpy.ndarray` object
        Depending on the input parameters, an array with n dimensions
        containing the pair-wise differences between `array1` and `array2` over
        the given `axis`.

    Examples
    --------
    Using two matrices returns the pair-wise differences in row-vectors:

        >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        >>> diff(mat1, mat2)
        array([[[-3., -3., -3.],
                [-6., -6., -6.]],
        <BLANKLINE>
               [[ 0.,  0.,  0.],
                [-3., -3., -3.]]])


    Setting `axis` to 1 returns the pair-wise differences in column-vectors:

        >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mat2 = np.array([[4, 5, 6], [7, 8, 9]])
        >>> diff(mat1, mat2, axis=1)
        array([[[-3., -3.],
                [-4., -4.],
                [-5., -5.]],
        <BLANKLINE>
               [[-2., -2.],
                [-3., -3.],
                [-4., -4.]],
        <BLANKLINE>
               [[-1., -1.],
                [-2., -2.],
                [-3., -3.]]])


    Only using a single matrix returns the pair-wise differences in row-vectors
    in that matrix (either flattened or not):

        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> diff(mat, flatten=True)
        array([[-3., -3., -3.]])
        >>> diff(mat, flatten=False)
        array([[[ 0.,  0.,  0.],
                [-3., -3., -3.]],
        <BLANKLINE>
               [[ 3.,  3.,  3.],
                [ 0.,  0.,  0.]]])


    Using a matrix and a vector returns the pair-wise differences in
    row-vectors:

        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> vec = np.array([7, 8, 9])
        >>> diff(mat, vec)
        array([[-6, -6, -6],
               [-3, -3, -3]])


    Using two vectors returns the pair-wise differences in scalars:

        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([4, 5, 6])
        >>> diff(vec1, vec2)
        array([[-3., -4., -5.],
               [-2., -3., -4.],
               [-1., -2., -3.]])

    """

    # If array2 is not provided, both arrays are the same
    if array2 is None:
        # Make sure that input is a numpy array
        array1 = np.asarray(array1)

        # Check if a scalar has been provided and act accordingly
        if(array1.ndim == 0):
            return(0)

        # Swap axes in array to put the given axis as the first axis
        try:
            array1 = np.moveaxis(array1, axis, 0)
        except Exception as error:
            raise InputError("Input argument 'axis' is invalid (%s)!"
                             % (error))
        else:
            # Obtain the dimensionality and axis-length
            n_dim = array1.ndim
            len_axis = array1.shape[0]

        # If only unique pair-wise differences are requested
        if flatten:
            # Obtain the shape of the resulting array and initialize it
            n_diff = len_axis*(len_axis-1)//2
            if(n_dim == 1):
                diff_shape = [n_diff]
            else:
                diff_shape = np.concatenate([[n_diff], array1.shape[1:n_dim]])
            diff_array = np.zeros(diff_shape)

            # Initialize empty variable holding the distance in index of last i
            dist = 0

            # Fill array
            for i in range(len_axis):
                diff_array[dist:dist+len_axis-i-1] = array1[i]-array1[i+1:]
                dist += len_axis-i-1

            # Return it
            return(diff_array)

        # If all difference are requested
        else:
            # Obtain the shape of the resulting array and initialize it
            diff_shape = np.concatenate([[len_axis], array1.shape])
            diff_array = np.zeros(diff_shape)

            # Fill array
            for i in range(len_axis):
                diff_array[i] = array1[i]-array1

            # Return it
            return(diff_array)

    # If array2 is provided, both arrays are different
    else:
        # Make sure that inputs are numpy arrays
        array1 = np.asarray(array1)
        array2 = np.asarray(array2)

        # Get number of dimensions
        n_dim1 = array1.ndim
        n_dim2 = array2.ndim

        # Check if both arrays are scalars and act accordingly
        if(n_dim1 == n_dim2 == 0):
            return(array1-array2)

        # If both arrays have the same number of dimensions
        if(n_dim1 == n_dim2):
            # Swap axes in arrays to put the given axis as the first axis
            try:
                array1 = np.moveaxis(array1, axis, 0)
                array2 = np.moveaxis(array2, axis, 0)
            except Exception as error:
                raise InputError("Input argument 'axis' is invalid (%s)!"
                                 % (error))
            else:
                # Obtain axis-length
                len_axis1 = array1.shape[0]

            # Check if the length of all other axes are the same
            if(array1.shape[1:] != array2.shape[1:]):
                raise ShapeError("Input arguments 'array1' and 'array2' do not"
                                 " have the same axes lengths: %s != %s"
                                 % (array1.shape[1:], array2.shape[1:]))

            # Obtain the shape of the resulting array and initialize it
            diff_shape = np.concatenate([[len_axis1], array2.shape])
            diff_array = np.zeros(diff_shape)

            # Fill array
            for i in range(len_axis1):
                diff_array[i] = array1[i]-array2

            # Return it
            return(diff_array)

        # If the arrays have different number of dimensions
        else:
            # If second array is bigger than first, swap them
            if(n_dim1 < n_dim2):
                # Swap arrays
                temp_array = array1
                array1 = array2
                array2 = temp_array

                # Swap ndims
                temp_ndim = n_dim1
                n_dim1 = n_dim2
                n_dim2 = temp_ndim

                # Save that arrays were swapped
                sign = -1
            else:
                sign = 1

            # Swap axes in the bigger array to put the given axis as first axis
            try:
                array1 = np.moveaxis(array1, axis, 0)
            except Exception as error:
                raise InputError("Input argument 'axis' is invalid (%s)!"
                                 % (error))

            # Check if the length of all other axes are the same
            if(array1.shape[1:n_dim1] != array2.shape):
                args = ((array1.shape[1:n_dim1], array2.shape) if(sign == 1)
                        else (array2.shape, array1.shape[1:n_dim1]))
                raise ShapeError("Input arguments 'array1' and 'array2' do"
                                 " not have the same axes lengths: %s != "
                                 "%s" % args)
            else:
                # Return difference array
                return(sign*(array1-array2))


# This function returns the intersection between two NumPy arrays
def intersect(array1, array2, axis=0, assume_unique=False):
    """
    Finds the intersection between given arrays `array1` and `array2` over
    provided `axis` and returns the unique elements that are both in `array1`
    and `array2`.

    This is an nD-version of NumPy's :func:`~numpy.intersect1d` function.

    Parameters
    ----------
    array1 : array_like
        Input array.
    array2 : array_like
        Comparison array with same shape as `array1` except in given `axis`.

    Optional
    --------
    axis : int or None. Default: 0
        Axis over which elements must be checked in both arrays. A negative
        value counts from the last to the first axis.
        If *None*, both arrays are flattened first (this is the functionality
        of :func:`~numpy.intersect1d`).
    assume_unique : bool. Default: False
        Whether to assume that the elements in both arrays are unique, which
        can speed up the calculation.

    Returns
    -------
    intersect_array : :obj:`~numpy.ndarray` object
        Array containing the unique elements found both in `array1` and
        `array2` over given `axis`.

    Example
    -------
    >>> array1 = np.array([[1, 2], [1, 3], [2, 1]])
    >>> array2 = np.array([[1, 2], [1, 3]])
    >>> intersect(array1, array2)
    array([[1, 2], [1, 3]])

    """

    # Check if axis is None
    if axis is None:
        # If so, use NumPy's intersect1d function
        return(np.intersect1d(array1, array2, assume_unique, False))

    # If assume_unique is False, make sure that the arrays are unique
    if not assume_unique:
        array1 = np.unique(array1, axis=axis)
        array2 = np.unique(array2, axis=axis)

    # Obtain which elements in array1 are in array2
    bool_array = isin(array1, array2, axis, True, invert=False)

    # Obtain the array with the unique elements of both arrays
    intersect_array = np.compress(bool_array, array1, axis)

    # Return it
    return(intersect_array)


# This function returns which elements of array1 are in array2
def isin(array1, array2, axis=0, assume_unique=False, invert=False):
    """
    Checks over the provided `axis` which elements of given `array1` are also
    in given `array2` and returns it.

    This is an nD-version of NumPy's :func:`~numpy.isin` function.

    Parameters
    ----------
    array1 : array_like
        Input array.
    array2 : array_like
        Comparison array with same shape as `array1` except in given `axis`.

    Optional
    --------
    axis : int or None. Default: 0
        Axis over which elements must be checked in both arrays. A negative
        value counts from the last to the first axis.
        If *None*, both arrays are compared element-wise (this is the
        functionality of :func:`~numpy.isin`).
    assume_unique : bool. Default: False
        Whether to assume that the elements in both arrays are unique, which
        can speed up the calculation.
    invert : bool. Default: False
        Whether to invert the returned boolean values. If *True*, the values in
        `bool_array` are as if calculating ``array1 not in array2``.

    Returns
    -------
    bool_array : :obj:`~numpy.ndarray` object of bool
        Bool array containing the elements found in `array1` that are in
        `array2` over given `axis`.

    Example
    -------
    >>> array1 = np.array([[1, 2], [1, 3], [2, 1]])
    >>> array2 = np.array([[1, 2], [1, 3]])
    >>> isin(array1, array2)
    array([True, True, False])

    """

    # Check if axis is None
    if axis is None:
        # If so, use NumPy's isin function
        return(np.isin(array1, array2, assume_unique, invert))

    # Make sure that given arrays are NumPy arrays
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)

    # Make sure that 'axis' is the first axis of both arrays
    try:
        array1 = np.moveaxis(array1, axis, 0)
        array2 = np.moveaxis(array2, axis, 0)
    except Exception as error:
        raise InputError("Input argument 'axis' is invalid (%s)!" % (error))

    # Convert arrays to lists
    list1 = array1.tolist()
    list2 = array2.tolist()

    # Determine what values in list1 are in list2
    bool_list = [element in list2 for element in list1]

    # Convert bool_list to bool_array
    bool_array = np.array(bool_list, dtype=bool)

    # Return it
    return(~bool_array if invert else bool_array)


# This function rotates a given array around a specified axis
def rot90(array, axes=(0, 1), rot_axis='center', n_rot=1):
    """
    Rotates the given `array` by 90 degrees around the point `rot_axis` in the
    given `axes`. This function is different from NumPy's :func:`~numpy.rot90`
    function in that every column (2nd axis) defines a different dimension
    instead of every individual axis.

    Parameters
    ----------
    array : 2D array_like
        Array with shape [`n_pts`, `n_dim`] with `n_pts` the number of points
        and `n_dim` the number of dimensions. Requires: `n_dim` > 1.

    Optional
    --------
    axes : 1D array_like with 2 ints. Default: (0, 1)
        Array containing the axes defining the rotation plane. Rotation is from
        the first axis towards the second. Can be omitted if `rot_axis` has
        length `n_dim`.
    rot_axis : 1D array_like of length 2/`n_dim` or 'center'. Default: 'center'
        If 'center', the rotation axis is chosen in the center of the minimum
        and maximum values found in the given `axes`.
        If 1D array of length 2, the rotation axis is chosen around the given
        values in the given `axes`.
        If 1D array of length `n_dim`, the rotation axis is chosen around the
        first two non-zero values.
    n_rot : int. Default: 1
        Number of times to rotate `array` by 90 degrees.

    Returns
    -------
    array_rot : 2D :obj:`~numpy.ndarray` object
        Array with shape [`n_pts`, `n_dim`] that has been rotated by 90 degrees
        `n_rot` times.

    Examples
    --------
    Using an array with just two dimensions:

        >>> array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        >>> rot90(array)
        array([[ 1.  ,  0.75],
               [ 0.  ,  0.25],
               [ 0.25,  1.  ],
               [ 0.75,  0.  ]])


    Using the same array, but rotating it around a different point:

        >>> array = np.array([[0.75, 0], [0.25, 1], [1, 0.75], [0, 0.25]])
        >>> rot90(array, rot_axis=[0.2, 0.7])
        array([[ 0.9 ,  1.25],
               [-0.1 ,  0.75],
               [ 0.15,  1.5 ],
               [ 0.65,  0.5 ]])

    """

    # Make sure that array is a numpy array
    array = np.asarray(array)

    # Check if array is indeed two-dimensional and obtain the lengths
    if(array.ndim != 2):
        raise ShapeError("Input argument 'array' must be two-dimensional!")
    else:
        n_pts, n_dim = array.shape

    # Check axes
    axes = np.asarray(axes)
    if(axes.ndim == 1 and axes.shape[0] == 2 and (axes < n_dim).all()):
        pass
    else:
        raise InputError("Input argument 'axes' has invalid shape or values!")

    # Check what rot_axis is and act accordingly
    if(rot_axis == 'center'):
        rot_axis = np.zeros(2)
        rot_axis[0] =\
            abs(np.max(array[:, axes[0]])+np.min(array[:, axes[0]]))/2
        rot_axis[1] =\
            abs(np.max(array[:, axes[1]])+np.min(array[:, axes[1]]))/2
    elif(isinstance(rot_axis, str)):
        raise ValueError("Input argument 'rot_axis' can only have 'center' as"
                         " a string value!")
    else:
        rot_axis = np.asarray(rot_axis)
        if(rot_axis.ndim == 1 and rot_axis.shape[0] == 2):
            pass
        elif(rot_axis.ndim == 1 and rot_axis.shape[0] == n_dim):
            axes = []
            for i in range(n_dim):
                if(rot_axis[i] != 0):
                    axes.append(i)
                if(len(axes) == 2):
                    break
            else:
                raise ValueError("Input argument 'rot_axis' does not have two "
                                 "non-zero values!")
            rot_axis = rot_axis[axes]
        else:
            raise ShapeError("Input argument 'rot_axis' has invalid shape!")

    # Calculate the rotated matrix
    array_rot = array.copy()
    if(n_rot % 4 == 0):
        return(array_rot)
    elif(n_rot % 4 == 1):
        array_rot[:, axes[0]] = rot_axis[0]+rot_axis[1]-array[:, axes[1]]
        array_rot[:, axes[1]] = rot_axis[1]-rot_axis[0]+array[:, axes[0]]
    elif(n_rot % 4 == 2):
        array_rot[:, axes[0]] = 2*rot_axis[0]-array[:, axes[0]]
        array_rot[:, axes[1]] = 2*rot_axis[1]-array[:, axes[1]]
    elif(n_rot % 4 == 3):
        array_rot[:, axes[0]] = rot_axis[0]-rot_axis[1]+array[:, axes[1]]
        array_rot[:, axes[1]] = rot_axis[1]+rot_axis[0]-array[:, axes[0]]
    else:
        raise InputError("Input argument 'n_rot' is invalid!")

    # Return it
    return(array_rot)


# This function returns the difference between two NumPy arrays
def setdiff(array1, array2, axis=0, assume_unique=False):
    """
    Finds the set difference between given arrays `array1` and `array2` over
    provided `axis` and returns the unique elements in `array1` that are not in
    `array2`.

    This is an nD-version of NumPy's :func:`~numpy.setdiff1d` function.

    Parameters
    ----------
    array1 : array_like
        Input array.
    array2 : array_like
        Comparison array with same shape as `array1` except in given `axis`.

    Optional
    --------
    axis : int or None. Default: 0
        Axis over which elements must be checked in both arrays. A negative
        value counts from the last to the first axis.
        If *None*, both arrays are flattened first (this is the functionality
        of :func:`~numpy.setdiff1d`).
    assume_unique : bool. Default: False
        Whether to assume that the elements in both arrays are unique, which
        can speed up the calculation.

    Returns
    -------
    diff_array : :obj:`~numpy.ndarray` object
        Array containing the unique elements found in `array1` but not in
        `array2` over given `axis`.

    Example
    -------
    >>> array1 = np.array([[1, 2], [1, 3], [2, 1]])
    >>> array2 = np.array([[1, 2], [1, 3]])
    >>> setdiff(array1, array2)
    array([[2, 1]])

    """

    # Check if axis is None
    if axis is None:
        # If so, use NumPy's setdiff1d function
        return(np.setdiff1d(array1, array2, assume_unique))

    # If assume_unique is False, make sure that the arrays are unique
    if not assume_unique:
        array1 = np.unique(array1, axis=axis)
        array2 = np.unique(array2, axis=axis)

    # Obtain which elements in array1 are not in array2
    bool_array = isin(array1, array2, axis, True, invert=True)

    # Obtain the array with the unique elements of array1
    diff_array = np.compress(bool_array, array1, axis)

    # Return it
    return(diff_array)


# This function returns the exclusive-or between two NumPy arrays
def setxor(array1, array2, axis=0, assume_unique=False):
    """
    Finds the set exclusive-or between given arrays `array1` and `array2` over
    provided `axis` and returns the unique elements that are in either `array1`
    or `array2` (but not both).

    This is an nD-version of NumPy's :func:`~numpy.setxor1d` function.

    Parameters
    ----------
    array1 : array_like
        Input array.
    array2 : array_like
        Comparison array with same shape as `array1` except in given `axis`.

    Optional
    --------
    axis : int or None. Default: 0
        Axis over which elements must be checked in both arrays. A negative
        value counts from the last to the first axis.
        If *None*, both arrays are flattened first (this is the functionality
        of :func:`~numpy.setxor1d`).
    assume_unique : bool. Default: False
        Whether to assume that the elements in both arrays are unique, which
        can speed up the calculation.

    Returns
    -------
    xor_array : :obj:`~numpy.ndarray` object
        Array containing the unique elements found in either `array1` or
        `array2` (but not both) over given `axis`.

    Example
    -------
    >>> array1 = np.array([[1, 2], [1, 3], [2, 1]])
    >>> array2 = np.array([[1, 2], [1, 3], [3, 1]])
    >>> setxor(array1, array2)
    array([[2, 1], [3, 1]])

    """

    # Check if axis is None
    if axis is None:
        # If so, use NumPy's setxor1d function
        return(np.setxor1d(array1, array2, assume_unique))

    # If assume_unique is False, make sure that the arrays are unique
    if not assume_unique:
        array1 = np.unique(array1, axis=axis)
        array2 = np.unique(array2, axis=axis)

    # Obtain the unique elements in array1 but not in array2
    xor_array1 = setdiff(array1, array2, axis, True)

    # Obtain the unique elements in array2 but not in array1
    xor_array2 = setdiff(array2, array1, axis, True)

    # Combine both arrays
    xor_array = np.concatenate([xor_array1, xor_array2], axis)

    # Sort the array
    xor_array = np.unique(xor_array, axis=axis)

    # Return it
    return(xor_array)


# This function sorts a 2D array in a specified order
def sort2D(array, axis=-1, order=None):
    """
    Sorts a 2D `array` over a given `axis` in the specified `order`. This
    function is different from NumPy's :func:`~numpy.sort` function in that it
    sorts over a given axis rather than along it, and the order can be given as
    integers rather than field strings.

    Parameters
    ----------
    array : 2D array_like
        Input array that requires sorting.

    Optional
    --------
    axis : int. Default: -1
        Axis over which to sort the elements. Default is to sort all elements
        over the last axis. A negative value counts from the last to the first
        axis.
    order : int, 1D array_like of int or None. Default: None
        The order in which the vectors in the given `axis` need to be sorted.
        Negative values count from the last to the first vector.
        If *None*, all vectors in the given `axis` are sorted individually.

    Returns
    -------
    array_sort : 2D :obj:`~numpy.ndarray` object
        Input `array` with its `axis` sorted in the specified `order`.

    Examples
    --------
    Sorting the column elements of a given 2D array with no order specified:

        >>> array = np.array([[0, 5, 1], [7, 4, 9], [3, 13, 6], [0, 1, 8]])
        >>> array
        array([[ 0,  5,  1],
               [ 7,  4,  9],
               [ 3, 13,  6],
               [ 0,  1,  8]])
        >>> sort2D(array)
        array([[ 0,  1,  1],
               [ 0,  4,  6],
               [ 3,  5,  8],
               [ 7, 13,  9]])


    Sorting the same array in only the first column:

        >>> sort2D(array, order=0)
        array([[ 0,  5,  1],
               [ 0,  1,  8],
               [ 3, 13,  6],
               [ 7,  4,  9]])


    Sorting all three columns in order:

        >>> sort2D(array, order=(0, 1, 2))
        array([[ 0,  1,  8],
               [ 0,  5,  1],
               [ 3, 13,  6],
               [ 7,  4,  9]])


    Sorting all three columns in a different order:

        >>> sort2D(array, order=(0, 2, 1))
        array([[ 0,  5,  1],
               [ 0,  1,  8],
               [ 3, 13,  6],
               [ 7,  4,  9]])

    """

    # Make sure that input array is a numpy array
    array = np.array(array)

    # Check if array is indeed 2D
    if(array.ndim != 2):
        raise ShapeError("Input argument 'array' must be two-dimensional!")
    else:
        # Obtain the number of vectors along the given axis
        try:
            n_vec = array.shape[axis]
        except Exception as error:
            raise InputError("Input argument 'axis' is invalid (%s)!"
                             % (error))

    # Move the given axis to be the first axis
    array = np.moveaxis(array, axis, 0)

    # If order is given, transform it into an array
    if order is not None:
        order = np.array(order, ndmin=1)

    # Check what order is given and act accordingly
    if order is None:
        array.sort(axis=-1)
    elif not(((-n_vec <= order)*(order < n_vec)).all()):
        raise ValueError("Input argument 'order' contains values that are "
                         "out of bounds!")
    else:
        for i in reversed(order):
            array = array[:, np.argsort(array[i], kind='mergesort')]

    # Return the resulting array back after transforming its axes back
    return(np.moveaxis(array, 0, axis))


# This function calculates the conjugate transpose of an array
def transposeC(array, axes=None):
    """
    Returns the (conjugate) transpose of the input `array`.

    Parameters
    ----------
    array : array_like
        Input array that needs to be transposed.

    Optional
    --------
    axes : 1D array_like of int or None. Default: None
        If *None*, reverse the dimensions.
        Else, permute the axes according to the values given.

    Returns
    -------
    array_t : :obj:`~numpy.ndarray` object
        Input `array` with its axes transposed.

    Examples
    --------
    Using an array with only real values returns its transposed variant:

        >>> array = np.array([[1, 2.5], [3.5, 5]])
        >>> array
        array([[ 1. ,  2.5],
               [ 3.5,  5. ]])
        >>> transposeC(array)
        array([[ 1. ,  3.5],
               [ 2.5,  5. ]])


    And using an array containing complex values returns its conjugate
    transposed:

        >>> array = np.array([[1, -2+4j], [7.5j, 0]])
        >>> array
        array([[ 1.+0.j , -2.+4.j ],
               [ 0.+7.5j,  0.+0.j ]])
        >>> transposeC(array)
        array([[ 1.-0.j ,  0.-7.5j],
               [-2.-4.j ,  0.-0.j ]])

    """

    # Take the transpose of the conjugate or the input array and return it
    return(np.transpose(np.conjugate(array), axes))


# This function returns the union between two NumPy arrays
def union(array1, array2, axis=0):
    """
    Finds the union between given arrays `array1` and `array2` over provided
    `axis` and returns the unique elements in `array1` and `array2`.

    This is an nD-version of NumPy's :func:`~numpy.union1d` function.

    Parameters
    ----------
    array1 : array_like
        Input array.
    array2 : array_like
        Comparison array with same shape as `array1` except in given `axis`.

    Optional
    --------
    axis : int or None. Default: 0
        Axis over which elements must be checked in both arrays. A negative
        value counts from the last to the first axis.
        If *None*, both arrays are flattened first (this is the functionality
        of :func:`~numpy.union1d`).

    Returns
    -------
    union_array : :obj:`~numpy.ndarray` object
        Sorted array containing the unique elements found in `array1` and
        `array2` over given `axis`.

    Example
    -------
    >>> array1 = np.array([[1, 2], [1, 3], [3, 1]])
    >>> array2 = np.array([[1, 2], [1, 3], [2, 1]])
    >>> union(array1, array2)
    array([[1, 2], [1, 3], [2, 1], [3, 1]])

    """

    # Check if axis is None
    if axis is None:
        # If so, use NumPy's union1d function
        return(np.union1d(array1, array2))

    # Combine both arrays together
    union_array = np.concatenate([array1, array2], axis)

    # Obtain the unique elements in this array
    union_array = np.unique(union_array, axis=axis)

    # Return it
    return(union_array)
