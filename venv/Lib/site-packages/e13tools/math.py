# -*- coding: utf-8 -*-

"""
Math
====
Provides a collection of functions useful in various mathematical calculations.

"""


# %% IMPORTS
# Built-in imports
from functools import reduce
from math import factorial

# Package imports
import numpy as np
from numpy.linalg import cholesky, eigvals, LinAlgError, norm, svd

# e13Tools imports
from e13tools.core import ShapeError
from e13tools.numpy import transposeC

# All declaration
__all__ = ['gcd', 'is_PD', 'lcm', 'nCr', 'nearest_PD', 'nPr']


# %% FUNCTIONS
# This function calculates the greatest common divisor of a sequence
def gcd(*args):
    """
    Returns the greatest common divisor of the provided sequence of integers.

    Parameters
    ----------
    args : tuple of int
        Integers to calculate the greatest common divisor for.

    Returns
    -------
    gcd : int
        Greatest common divisor of input integers.

    Example
    -------
    >>> gcd(18, 60, 72, 138)
    6

    See also
    --------
    :func:`~lcm`
        Least common multiple for sequence of integers.

    """

    return(reduce(gcd_single, args))


# This function calculates the greatest common divisor of two integers
def gcd_single(a, b):
    """
    Returns the greatest common divisor of the integers `a` and `b` using
    Euclid's Algorithm [1]_.

    Parameters
    ----------
    a, b : int
        The two integers to calculate the greatest common divisor for.

    Returns
    -------
    gcd : int
        Greatest common divisor of `a` and `b`.

    Notes
    -----
    The calculation of the greatest common divisor uses Euclid's Algorithm [1]_
    with Lamé's improvements.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm

    Example
    -------
    >>> gcd_single(42, 56)
    14

    See also
    --------
    :func:`~gcd`
        Greatest common divisor for sequence of integers.

    :func:`~lcm`
        Least common multiple for sequence of integers.

    :func:`~core.lcm_single`
        Least common multiple for two integers.

    """

    while(b):
        a, b = b, a % b
    return(a)


# This function determines if a matrix is positive-definite
def is_PD(matrix):
    """
    Checks if `matrix` is positive-definite or not, by using the
    :func:`~numpy.linalg.cholesky` function. It is required for `matrix` to be
    Hermitian.

    Parameters
    ----------
    matrix : 2D array_like
        Matrix that requires checking.

    Returns
    -------
    out: bool
        *True* if `matrix` is positive-definite, *False* if it is not.

    Examples
    --------
    Using a real matrix that is positive-definite (like the identity matrix):

        >>> matrix = np.eye(3)
        >>> matrix
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
        >>> is_PD(matrix)
        True


    Using a real matrix that is not symmetric (Hermitian):

        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> matrix
        array([[1, 2],
               [3, 4]])
        >>> is_PD(matrix)
        Traceback (most recent call last):
            ...
        ValueError: Input argument 'matrix' must be Hermitian!


    Using a complex matrix that is positive-definite:

        >>> matrix = np.array([[4, 1.5+1j], [1.5-1j, 3]])
        >>> matrix
        array([[ 4.0+0.j,  1.5+1.j],
               [ 1.5-1.j,  3.0+0.j]])
        >>> is_PD(matrix)
        True

    See also
    --------
    :func:`~nearest_PD`
        Find the nearest positive-definite matrix to the input `matrix`.

    """

    # Make sure that matrix is a numpy array
    matrix = np.asarray(matrix)

    # Check if input is a matrix
    if(matrix.ndim != 2):
        raise ShapeError("Input argument 'matrix' must be two-dimensional!")
    else:
        rows, columns = matrix.shape

    # Check if matrix is a square
    if(rows != columns):
        raise ShapeError("Input argument 'matrix' has shape [%s, %s]. 'matrix'"
                         " must be a square matrix!" % (rows, columns))

    # Check if matrix is Hermitian
    if not np.allclose(transposeC(matrix), matrix):
        raise ValueError("Input argument 'matrix' must be Hermitian!")

    # Try to use Cholesky on matrix. If it fails,
    try:
        cholesky(matrix)
    except LinAlgError:
        return(False)
    else:
        return(True)


# This function calculates the least common multiple of a sequence
def lcm(*args):
    """
    Returns the least common multiple of the provided sequence of integers.
    If at least one integer is zero, the output will also be zero.

    Parameters
    ----------
    args : tuple of int
        Integers to calculate the least common multiple for.

    Returns
    -------
    lcm : int
        Least common multiple of input integers.

    Example
    -------
    >>> lcm(8, 9, 21)
    504

    See also
    --------
    :func:`~gcd`
        Greatest common divisor for sequence of integers.

    """

    return(reduce(lcm_single, args))


# This function calculates the least common multiple of two integers
def lcm_single(a, b):
    """
    Returns the least common multiple of the integers `a` and `b`.
    If at least one integer is zero, the output will also be zero.

    Parameters
    ----------
    a, b : int
        The two integers to calculate the least common multiple for.

    Returns
    -------
    lcm : int
        Least common multiple of `a` and `b`.

    Notes
    -----
    The least common multiple of two given integers :math:`a` and :math:`b` is
    given by

        .. math:: \\mathrm{lcm}(a, b)=\\frac{|a\\cdot b|}{\\mathrm{gcd}(a, b)},

    which can also be written as

        .. math:: \\mathrm{lcm}(a, b)=\\frac{|a|}{\\mathrm{gcd}(a, b)}\\cdot \
            |b|,

    with :math:`\\mathrm{gcd}` being the greatest common divisor.

    Example
    -------
    >>> lcm_single(6, 21)
    42

    See also
    --------
    :func:`~gcd`
        Greatest common divisor for sequence of integers.

    :func:`~core.gcd_single`
        Greatest common divisor for two integers.

    :func:`~lcm`
        Least common multiple for sequence of integers.

    """

    return(0 if(a == 0 or b == 0) else (abs(a)//gcd_single(a, b))*abs(b))


# This function calculates the number of unordered arrangements
def nCr(n, r, repeat=False):
    """
    For a given set S of `n` elements, returns the number of unordered
    arrangements ("combinations") of length `r` one can make with S.
    Returns zero if `r` > `n` and `repeat` is *False*.

    Parameters
    ----------
    n : int
        Number of elements in the set S.
    r : int
        Number of elements in the sub-set of set S.

    Optional
    --------
    repeat : bool. Default: False
        If *False*, each element in S can only be chosen once.
        If *True*, they can be chosen more than once.

    Returns
    -------
    n_comb : int
        Number of "combinations" that can be made with S.

    Examples
    --------
    >>> nCr(4, 2)
    6


    >>> nCr(4, 2, repeat=True)
    10


    >>> nCr(2, 4, repeat=True)
    5


    >>> nCr(2, 4)
    0

    See also
    --------
    :func:`~nPr`
        Returns the number of ordered arrangements.

    """

    # Check if repeat is True or not and act accordingly
    if(r == 0):
        return(1)
    elif(r == 1):
        return(n)
    elif repeat:
        return(factorial(n+r-1)//(factorial(r)*factorial(n-1)))
    elif(r == n-1):
        return(n)
    elif(r == n):
        return(1)
    elif(r > n):
        return(0)
    else:
        return(factorial(n)//(factorial(r)*factorial(n-r)))


# This function converts a given matrix to its nearest PD variant
def nearest_PD(matrix):
    """
    Find the nearest positive-definite matrix to the input `matrix`.

    Parameters
    ----------
    matrix : 2D array_like
        Input matrix that requires its nearest positive-definite variant.

    Returns
    -------
    mat_PD : 2D :obj:`~numpy.ndarray` object
        The nearest positive-definite matrix to the input `matrix`.

    Notes
    -----
    This is a Python port of John D'Errico's *nearestSPD* code [1]_, which is a
    MATLAB implementation of Higham (1988) [2]_.

    According to Higham (1988), the nearest positive semi-definite matrix in
    the Frobenius norm to an arbitrary real matrix :math:`A` is shown to be

        .. math:: \\frac{B+H}{2},

    with :math:`H` being the symmetric polar factor of

        .. math:: B=\\frac{A+A^T}{2}.

    On page 2, the author mentions that all matrices :math:`A` are assumed to
    be real, but that the method can be very easily extended to the complex
    case. This can indeed be done easily by taking the conjugate transpose
    instead of the normal transpose in the formula on the above.

    References
    ----------
    .. [1] \
        https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    .. [2] N.J. Higham, "Computing a Nearest Symmetric Positive Semidefinite
           Matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Examples
    --------
    Requesting the nearest PD variant of a matrix that is already PD results
    in it being returned immediately:

        >>> matrix = np.eye(3)
        >>> matrix
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
        >>> is_PD(matrix)
        True
        >>> nearest_PD(matrix)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])


    Using a real non-PD matrix results in it being transformed into an
    PD-matrix:

        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> matrix
        array([[1, 2],
               [3, 4]])
        >>> is_PD(matrix)
        Traceback (most recent call last):
            ...
        ValueError: Input argument 'matrix' must be Hermitian!
        >>> mat_PD = nearest_PD(matrix)
        >>> mat_PD
        array([[ 1.31461828,  2.32186616],
               [ 2.32186616,  4.10085767]])
        >>> is_PD(mat_PD)
        True


    Using a complex non-PD matrix converts it into the nearest complex
    PD-matrix:

        >>> matrix = np.array([[4, 2+1j], [1+3j, 3]])
        >>> matrix
        array([[ 4.+0.j,  2.+1.j],
               [ 1.+3.j,  3.+0.j]])
        >>> mat_PD = nearest_PD(matrix)
        >>> mat_PD
        array([[ 4.0+0.j,  1.5-1.j],
               [ 1.5+1.j,  3.0+0.j]])
        >>> is_PD(mat_PD)
        True

    See also
    --------
    :func:`~is_PD`
        Checks if `matrix` is positive-definite or not.

    """

    # Make sure that matrix is a numpy array
    matrix = np.asarray(matrix)

    # Check if input is a matrix
    if(matrix.ndim != 2):
        raise ShapeError("Input argument 'matrix' must be two-dimensional!")
    else:
        rows, columns = matrix.shape

    # Check if matrix is a square
    if(rows != columns):
        raise ShapeError("Input argument 'matrix' has shape [%s, %s]. 'matrix'"
                         " must be a square matrix!" % (rows, columns))

    # Check if matrix is not already positive-definite
    try:
        is_PD(matrix)
    except ValueError:
        pass
    else:
        if is_PD(matrix):
            return(matrix)

    # Make sure that the matrix is Hermitian
    mat_H = (matrix+transposeC(matrix))/2

    # Perform singular value decomposition
    _, S, VH = svd(mat_H)

    # Compute the symmetric polar factor of mat_H
    spf = np.dot(transposeC(VH), np.dot(np.diag(S), VH))

    # Obtain the positive-definite matrix candidate
    mat_PD = (mat_H+spf)/2

    # Ensure that mat_PD is Hermitian
    mat_PD = (mat_PD+transposeC(mat_PD))/2

    # Check if mat_PD is in fact positive-definite
    if is_PD(mat_PD):
        return(mat_PD)

    # If it is not, change it very slightly to make it positive-definite
    In = np.eye(rows)
    k = 1
    spacing = np.spacing(norm(matrix))
    while not is_PD(mat_PD):
        min_eig_val = np.min(np.real(eigvals(mat_PD)))
        mat_PD += In*(-1*min_eig_val*pow(k, 2)+spacing)
        k += 1
    else:
        return(mat_PD)


# This function calculates the number of ordered arrangements
def nPr(n, r, repeat=False):
    """
    For a given set S of `n` elements, returns the number of ordered
    arrangements ("permutations") of length `r` one can make with S.
    Returns zero if `r` > `n` and `repeat` is *False*.

    Parameters
    ----------
    n : int
        Number of elements in the set S.
    r : int
        Number of elements in the sub-set of set S.

    Optional
    --------
    repeat : bool. Default: False
        If *False*, each element in S can only be chosen once.
        If *True*, they can be chosen more than once.

    Returns
    -------
    n_perm : int
        Number of "permutations" that can be made with S.

    Examples
    --------
    >>> nPr(4, 2)
    12


    >>> nPr(4, 2, repeat=True)
    16


    >>> nPr(2, 4, repeat=True)
    16


    >>> nPr(2, 4)
    0

    See also
    --------
    :func:`~nCr`
        Returns the number of unordered arrangements.

    """

    # Check if repeat is True or not and act accordingly
    if(r == 0):
        return(1)
    elif(r == 1):
        return(n)
    elif repeat:
        return(pow(n, r))
    elif(r > n):
        return(0)
    else:
        return(factorial(n)//factorial(n-r))
