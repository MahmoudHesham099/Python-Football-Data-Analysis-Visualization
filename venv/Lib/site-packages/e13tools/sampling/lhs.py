# -*- coding: utf-8 -*-

"""
LHS
===
Provides a Latin Hypercube Sampling method.

Available functions
-------------------
:func:`~lhd`
    Generates a Latin Hypercube Design of `n_sam` samples, each with `n_val`
    values. Method for choosing the 'best' Latin Hypercube Design depends on
    the `method` and `criterion` that are used.

"""


# %% IMPORTS
# Package imports
import numpy as np
from numpy.random import choice, permutation, rand

# e13Tools imports
from e13tools.core import ShapeError
from e13tools.math import nCr
from e13tools.numpy import diff

# All declaration
__all__ = ['lhd']


# %% FUNCTIONS
# TODO: Check if MPI is possible
def lhd(n_sam, n_val, val_rng=None, method='random', criterion=None,
        iterations=1000, get_score=False, quickscan=True, constraints=None):
    """
    Generates a Latin Hypercube Design of `n_sam` samples, each with `n_val`
    values. Method for choosing the 'best' Latin Hypercube Design depends on
    the `method` and `criterion` that are used.

    Parameters
    ----------
    n_sam : int
        The number of samples to generate.
    n_val : int
        The number of values in a single sample.

    Optional
    --------
    val_rng : 2D array_like or None. Default: None
        Array defining the lower and upper limits of every value in a sample.
        Requires: numpy.shape(val_rng) = (`n_val`, 2).
        If *None*, output is normalized.
    method : {'random'; 'fixed'; 'center'}. Default: 'random'
        String specifying the method used to construct the Latin Hypercube
        Design. See ``Notes`` for more details.
        If `n_sam` == 1 or `n_val` == 1, `method` is set to the closest
        corresponding method if necessary.
    criterion : float, {'maximin'; 'correlation'; 'multi'} or None. \
        Default: None
        Float or string specifying the criterion the Latin Hypercube Design has
        to satisfy or *None* for no criterion. See ``Notes`` for more details.
        If `n_sam` == 1 or `n_val` == 1, `criterion` is set to the closest
        corresponding criterion if necessary.
    iterations : int. Default: 1000
        Number of iterations used for the criterion algorithm.
    get_score : bool. Default: False
        If *True*, the normalized maximin, correlation and multi scores are
        also returned if a criterion is used.
    quickscan : bool. Default: True
        If *True*, a faster but less precise algorithm will be used for the
        criteria.
    constraints : 2D array_like or None. Default: None
        If `constraints` is not empty and `criterion` is not *None*, `sam_set`
        + `constraints` will satisfy the given criterion instead of `sam_set`.
        Providing this argument when `criterion` is *None* will discard it.
        **WARNING**: If `constraints` is not a 'fixed' or 'center' lay-out LHD,
        the output might contain errors.

    Returns
    -------
    sam_set : 2D :obj:`~numpy.ndarray` object
        Sample set array of shape [`n_sam`, `n_val`].

    Notes
    -----
    The 'method' argument specifies the way in which the values should be
    distributed within the value intervals.

    The following methods can be used:

    ======== ===================================
    method   interval lay-out
    ======== ===================================
    'random' Values are randomized
    'fixed'  Values are fixed to maximize spread
    'center' Values are centered
    'r'      Same as 'random'
    'f'      Same as 'fixed'
    'c'      Same as 'center'
    ======== ===================================

    The 'fixed' method chooses values in such a way, that the distance between
    the values is maxed out.


    The 'criterion' argument specifies how much priority should be given to
    maximizing the minimum distance and minimizing the correlation between
    samples. Strings specify basic priority cases, while a value between 0 and
    1 specifies a custom case.

    The following criteria can be used (last column shows the equivalent
    float value):

    ============= ==================================================== ======
    criterion     effect/priority                                      equiv
    ============= ==================================================== ======
    None          No priority                                          --
    'maximin'     Maximum priority for maximizing the minimum distance 0.0
    'correlation' Maximum priority for minimizing the correlation      1.0
    'multi'       Equal priority for both                              0.5
    [0, 1]        Priority is given according to value provided        --
    ============= ==================================================== ======

    Examples
    --------
    Latin Hypercube with 5 samples with each 2 random, fixed or centered
    values:

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> lhd(5, 2, method='random')
        array([[ 0.34303787,  0.55834501],
               [ 0.70897664,  0.70577898],
               [ 0.88473096,  0.19273255],
               [ 0.1097627 ,  0.91360891],
               [ 0.52055268,  0.2766883 ]])
        >>> lhd(5, 2, method='fixed')
        array([[ 0.5 ,  0.75],
               [ 0.25,  0.25],
               [ 0.  ,  1.  ],
               [ 0.75,  0.5 ],
               [ 1.  ,  0.  ]])
        >>> lhd(5, 2, method='center')
        array([[ 0.1,  0.9],
               [ 0.9,  0.5],
               [ 0.5,  0.7],
               [ 0.3,  0.3],
               [ 0.7,  0.1]])


    Latin Hypercube with 4 samples, 3 values in a specified value range:

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> val_rng = [[0, 2], [1, 4], [0.3, 0.5]]
        >>> lhd(4, 3, val_rng=val_rng)
        array([[ 1.30138169,  2.41882975,  0.41686981],
               [ 0.27440675,  1.32819041,  0.48240859],
               [ 1.77244159,  3.53758114,  0.39180394],
               [ 0.85759468,  3.22274707,  0.31963924]])


    Latin Hypercubes can also be created by specifying a criterion with either
    a string or a normalized float. The strings identify basic float values.

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> lhd(4, 3, method='fixed', criterion=0)
        array([[ 0.66666667,  0.        ,  0.66666667],
               [ 1.        ,  0.66666667,  0.        ],
               [ 0.33333333,  1.        ,  1.        ],
               [ 0.        ,  0.33333333,  0.33333333]])
        >>> np.random.seed(0)
        >>> lhd(4, 3, method='fixed', criterion='maximin')
        array([[ 0.66666667,  0.        ,  0.66666667],
               [ 1.        ,  0.66666667,  0.        ],
               [ 0.33333333,  1.        ,  1.        ],
               [ 0.        ,  0.33333333,  0.33333333]])

    """

    # Make sure that if val_rng is given, that it is valid
    if val_rng is not None:
        # If val_rng is 1D, convert it to 2D (expected for 'n_val' = 1)
        val_rng = np.array(val_rng, ndmin=2)

        # Check if the given val_rng is in the correct shape
        if not(val_rng.shape == (n_val, 2)):
            raise ShapeError("'val_rng' has incompatible shape: %s != (%s, %s)"
                             % (val_rng.shape, n_val, 2))

    # TODO: Implement constraints method again!
    # Make sure that constraints is a numpy array
    if constraints is not None:
        constraints = np.array(constraints, ndmin=2)

    # Check the shape of 'constraints' and act accordingly
    if constraints is None:
        pass
    elif(constraints.shape[-1] == 0):
        # If constraints is empty, there are no constraints
        constraints = None
    elif(constraints.ndim != 2):
        # If constraints is not two-dimensional, it is invalid
        raise ShapeError("Constraints must be two-dimensional!")
    elif(constraints.shape[-1] == n_val):
        # If constraints has the same number of values, it is valid
        constraints = _extract_sam_set(constraints, val_rng)
    else:
        # If not empty and not right shape, it is invalid
        raise ShapeError("Constraints has incompatible number of values: "
                         "%s =! %s" % (np.shape(constraints)[1], n_val))

    # Check for cases in which some methods make no sense
    if(n_sam == 1 and method.lower() in ('fixed', 'f')):
        method = 'center'
    elif(criterion is not None and method.lower() in ('random', 'r')):
        method = 'fixed'

    # Check for cases in which some criterions make no sense
    # If so, criterion will be changed to something useful
    if criterion is None:
        pass
    elif(n_sam == 1):
        criterion = None
    elif(n_val == 1 or n_sam == 2):
        criterion = None
    elif isinstance(criterion, (int, float)):
        if not(0 <= criterion <= 1):
            raise ValueError("Input argument 'criterion' can only have a "
                             "normalized value as a float value!")
    elif criterion.lower() not in ('maximin', 'correlation', 'multi'):
        raise ValueError("Input argument 'criterion' can only have {'maximin',"
                         " 'correlation', 'multi'} as string values!")

    # Pick correct lhs-method according to method
    if method.lower() in ('random', 'r'):
        sam_set = _lhd_random(n_sam, n_val)
    elif method.lower() in ('fixed', 'f'):
        sam_set = _lhd_fixed(n_sam, n_val)
    elif method.lower() in ('center', 'c'):
        sam_set = _lhd_center(n_sam, n_val)

    # Pick correct criterion
    if criterion is not None:
        multi_obj = Multi_LHD(sam_set, criterion, iterations, quickscan,
                              constraints)
        sam_set, mm_val, corr_val, multi_val = multi_obj()

    # If a val_rng was given, scale sam_set to this range
    if val_rng is not None:
        # Scale sam_set according to val_rng
        sam_set = val_rng[:, 0]+sam_set*(val_rng[:, 1]-val_rng[:, 0])

    if get_score and criterion is not None:
        return(sam_set, np.array([mm_val, corr_val, multi_val]))
    else:
        return(sam_set)


def _lhd_random(n_sam, n_val):
    # Generate the equally spaced intervals/bins
    bins = np.linspace(0, 1, n_sam+1)

    # Obtain lower and upper bounds of bins
    bins_low = bins[0:n_sam]
    bins_high = bins[1:n_sam+1]

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = permutation(bins_low+rand(n_sam)*(bins_high-bins_low))

    # Return sam_set
    return(sam_set)


def _lhd_fixed(n_sam, n_val):
    # Generate the maximally spaced values in every dimension
    val = np.linspace(0, 1, n_sam)

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = permutation(val)

    # Return sam_set
    return(sam_set)


def _lhd_center(n_sam, n_val):
    # Generate the equally spaced intervals/bins
    bins = np.linspace(0, 1, n_sam+1)

    # Obtain lower and upper bounds of bins
    bins_low = bins[0:n_sam]
    bins_high = bins[1:n_sam+1]

    # Capture centers of every bin
    center_num = (bins_low+bins_high)/2

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = permutation(center_num)

    # Return sam_set
    return(sam_set)


class Multi_LHD(object):
    def __init__(self, sam_set, criterion, iterations, quickscan, constraints):
        # Save all arguments as class attributes
        self.sam_set = sam_set
        self.iterations = iterations
        self.quickscan = quickscan
        self.n_sam, self.n_val = self.sam_set.shape

        # Combine constraints with sam_set
        if constraints is not None:
            self.n_sam_c, _ = constraints.shape
            self.sam_set = np.concatenate([self.sam_set, constraints], axis=0)
        else:
            self.n_sam_c = 0

        self.sam_set = self.sam_set*(self.n_sam-1)
        self.p = 15

        # Check criterion type and act accordingly
        if isinstance(criterion, (int, float)):
            self.importance = criterion
        elif(criterion.lower() == 'maximin'):
            self.importance = 0
        elif(criterion.lower() == 'correlation'):
            self.importance = 1
        elif(criterion.lower() == 'multi'):
            self.importance = 0.5

        # Obtain maximin-rank boundaries
        self._get_mm_bounds()

    def __call__(self):
        self._lhd_multi()
        return(self.sam_set_best[:self.n_sam]/(self.n_sam-1), self.mm_tot_best,
               np.sqrt(self.corr_tot_best), self.multi_val_best)
#        return(self.sam_set_best, self.mm_tot_best,
#               np.sqrt(self.corr_tot_best), self.multi_val_best)

    def _get_mm_bounds(self):
        # Calculate the p_dist of the provided sam_set
        p_dist_slice = self.n_sam_c*self.n_sam+nCr(self.n_sam, 2)
        p_dist = abs(diff(self.sam_set, axis=0, flatten=True))[:p_dist_slice]

        # Calculate the average distance between randomly chosen samples
        self.dist_avg = np.average(np.sum(p_dist, axis=-1))

        # TODO: Look at calculation of mm_lower once more
        # Calculate lower and upper boundaries of the maximin-rank
        self.mm_lower = pow(nCr(self.n_sam, 2), 1/self.p)/self.dist_avg
        p_dist_sort = np.sum(np.sort(p_dist, axis=0), axis=-1)

        # TODO: This has many exception cases, so maybe make an extra method?
        # If (due to constraints) any value in p_dist is zero, change p_dist in
        # such a way that it resembles the worst p_dist possible with no zeros
        if((p_dist_sort == 0).any()):
            values, counts = np.unique(p_dist_sort, return_counts=True)
#            p_dist_sort[:counts[0]] += values[1]/self.n_val
#            p_dist_sort[counts[0]:counts[1]] -= values[1]/self.n_val
            p_dist_sort[:counts[0]] += np.min(p_dist[p_dist != 0])
            p_dist_sort[counts[0]:counts[1]] -= np.min(p_dist[p_dist != 0])

        self.mm_upper = pow(np.sum(pow(p_dist_sort, -self.p)), 1/self.p)

    # TODO: If method is 'random', maybe randomly generate a new value in the
    # interval the values belong to after swapping. This way, the LHD will
    # actually be random
    def _lhd_multi(self):
        # Calculate the rank of this design
        sam_set = self.sam_set
        p_dist, mm_val = self._get_mm_val_init(sam_set)
        mm_tot = self._get_mm_tot(mm_val)
        corr_val = self._get_corr_val(sam_set)
        corr_tot = self._get_corr_tot(corr_val)
        multi_val = self._get_multi_val(corr_tot, mm_tot)

        # Save a copy of sam_set as sam_set_try
        sam_set_best = sam_set.copy()
        corr_tot_best = corr_tot
        mm_tot_best = mm_tot
        multi_val_best = multi_val

        # Maximize the minimum distance between points
        if self.quickscan:
            It = 0
            corr_val_try = corr_val
            corr_tot_try = corr_tot
            p_dist_try = p_dist
            mm_val_try = mm_val
            mm_tot_try = mm_tot
            multi_val_try = multi_val
            sam_set_try = sam_set.copy()
            while(It < self.iterations):
                # Randomly pick a column
                col = corr_val_try.argmax()
                rows = [mm_val_try.argmax()]
                rows_p = list(range(self.n_sam))
                rows_p.remove(rows[0])
                rows.append(choice(rows_p))

                # Swap elements
                val = sam_set_try[rows[0], col]
                sam_set_try[rows[0], col] = sam_set_try[rows[1], col]
                sam_set_try[rows[1], col] = val

                # Calculate the multi rank of sam_set_try
                p_dist_try, mm_val_try = self._get_mm_val(sam_set_try,
                                                          p_dist_try,
                                                          mm_val_try,
                                                          rows[0],
                                                          rows[1])
                mm_tot_try = self._get_mm_tot(mm_val_try)
                corr_val_try = self._get_corr_val(sam_set_try)
                corr_tot_try = self._get_corr_tot(corr_val_try)
                multi_val_try = self._get_multi_val(corr_tot_try, mm_tot_try)

                # If this rank is lower than current best rank, save sam_set
                if(multi_val_try < multi_val_best):
                    multi_val_best = multi_val_try
                    corr_tot_best = corr_tot_try
                    mm_tot_best = mm_tot_try
                    sam_set_best = sam_set_try.copy()
                    It = 0
                else:
                    It += 1

        else:
            t = 1
            flag = 1
            while(flag == 1):
                It = 0
                flag = 0
                while(It < self.iterations):
                    corr_val_try = corr_val
                    corr_tot_try = corr_tot
                    p_dist_try = p_dist
                    mm_val_try = mm_val
                    mm_tot_try = mm_tot
                    multi_val_try = multi_val
                    sam_set_try = sam_set.copy()

                    # Randomly pick a column
                    col = corr_val_try.argmax()
                    rows = [mm_val_try.argmax()]
                    rows_p = list(range(self.n_sam))
                    rows_p.remove(rows[0])
                    rows.append(choice(rows_p))

                    # Swap elements
                    val = sam_set_try[rows[0], col]
                    sam_set_try[rows[0], col] = sam_set_try[rows[1], col]
                    sam_set_try[rows[1], col] = val

                    # Calculate the multi rank of sam_set_try
                    p_dist_try, mm_val_try = self._get_mm_val(sam_set_try,
                                                              p_dist_try,
                                                              mm_val_try,
                                                              rows[0],
                                                              rows[1])
                    mm_tot_try = self._get_mm_tot(mm_val_try)
                    corr_val_try = self._get_corr_val(sam_set_try)
                    corr_tot_try = self._get_corr_tot(corr_val_try)
                    multi_val_try = self._get_multi_val(corr_tot_try,
                                                        mm_tot_try)

                    # If this rank is lower than the current rank, save sam_set
                    if(multi_val_try < multi_val or
                       (multi_val_try > multi_val and
                            rand() < np.exp((multi_val-multi_val_try)/t))):
                        corr_val = corr_val_try
                        corr_tot = corr_tot_try
                        p_dist = p_dist_try
                        mm_val = mm_val_try
                        mm_tot = mm_tot_try
                        multi_val = multi_val_try
                        sam_set = sam_set_try.copy()
                        flag = 1
                    # If this rank is lower than the current best rank, save it
                    if(multi_val_try < multi_val_best):
                        multi_val_best = multi_val_try
                        corr_tot_best = corr_tot_try
                        mm_tot_best = mm_tot_try
                        sam_set_best = sam_set_try.copy()
                        It = 0
                    else:
                        It += 1

                # Decrease temperature by 10%
                t *= 0.9

        # Return sam_set
        self.sam_set_best = sam_set_best
        self.corr_tot_best = corr_tot_best
        self.mm_tot_best = mm_tot_best
        self.multi_val_best = multi_val_best

    def _get_corr_val(self, sam_set):
        # Obtain the cross-correlation values between all columns in sam_set
        cross_corr = np.corrcoef(sam_set, rowvar=False)
        cross_corr[np.eye(self.n_val, dtype=int) == 1] = 0

        # Calculate the squared correlation values of all columns
        corr_val = np.sum(pow(cross_corr, 2), axis=-1)/(self.n_val-1)

        # Return it
        return(corr_val)

    def _get_mm_val_init(self, sam_set):
        # TODO: Check if masked arrays are really faster than lots of indexing
        # Calculate the pair-wise point distances
        masked_idx = np.diag_indices(self.n_sam, ndim=2)
        p_dist = np.sum(abs(diff(sam_set, axis=0, flatten=False)), axis=-1)
        p_dist = np.ma.array(p_dist, mask=False, hard_mask=True)
        p_dist.mask[masked_idx] = True
        p_dist.mask[self.n_sam:, self.n_sam:] = True

        # Create empty array containing the maximin values of all rows
        mm_val = np.zeros(self.n_sam)

        # Calculate maximin values
        for i, row_dist in enumerate(p_dist[:self.n_sam]):
            mm_val[i] = pow(np.sum(pow(row_dist, -self.p)), 1/self.p)

        # Return it
        return(p_dist, mm_val)

    def _get_mm_val(self, sam_set, p_dist_old, mm_val_old, r1, r2):
        # Create arrays containing new p_dist and mm_val
        p_dist = p_dist_old.copy()
        mm_val = np.zeros(self.n_sam)

        # Calculate new p_dist of row r1 and r2
        p_dist_r1 = np.sum(abs(diff(sam_set[r1], sam_set)), axis=-1)
        p_dist_r2 = np.sum(abs(diff(sam_set[r2], sam_set)), axis=-1)

        # Update p_dist and mm_val with newly calculated values
        p_dist[r1] = p_dist[:, r1] = p_dist_r1
        p_dist[r2] = p_dist[:, r2] = p_dist_r2
        iets1 = p_dist_r1[np.arange(self.n_sam+self.n_sam_c) != r1]
        iets2 = p_dist_r2[np.arange(self.n_sam+self.n_sam_c) != r2]
        if((iets1 == 0).any()):
            mm_val[r1] = np.infty
        else:
            mm_val[r1] = pow(np.sum(pow(iets1, -self.p)), 1/self.p)
        if((iets2 == 0).any()):
            mm_val[r2] = np.infty
        else:
            mm_val[r2] = pow(np.sum(pow(iets2, -self.p)), 1/self.p)

        # Create list containing only indices of unchanged rows
        idx = list(range(self.n_sam))
        idx.remove(r1)
        idx.remove(r2)

        # Update the mm_val of all unchanged rows
        mm_val[idx] = pow(pow(mm_val_old[idx], self.p) -
                          pow(p_dist_old[r1, idx], -self.p) -
                          pow(p_dist_old[r2, idx], -self.p) +
                          pow(p_dist_r1[idx], -self.p) +
                          pow(p_dist_r2[idx], -self.p),
                          1/self.p)

        # Return it
        return(p_dist, mm_val)

    def _get_mm_tot(self, mm_val):
        # Calculate the total mm_val
        mm_tot = pow(0.5*np.sum(pow(mm_val, self.p)), 1/self.p)
        return(((mm_tot-self.mm_lower)/(self.mm_upper-self.mm_lower)))

    def _get_corr_tot(self, corr_val):
        # Calculate the total corr_val
        return(np.sum(corr_val)/(self.n_val))

    def _get_multi_val(self, corr_tot, mm_tot):
        # Combine corr_tot and mm_tot to the multi_val
        return(self.importance*corr_tot+(1-self.importance)*mm_tot)


def _extract_sam_set(sam_set, val_rng):
    """
    Extracts the samples from `sam_set` that are within the given value
    ranges `val_rng`. Also extracts the two samples that are the closest to the
    given value ranges, but are outside of it.

    Parameters
    ----------
    sam_set : 2D array_like
        Sample set containing the samples that require extraction.
    val_rng : 2D array_like or None
        Array defining the lower and upper limits of every value in a sample.
        If *None*, output is normalized.

    Returns
    -------
    ext_sam_set : 2D array_like
        Sample set containing the extracted samples.

    """

    # Obtain number of values in number of samples
    n_sam, n_val = sam_set.shape

    # Check if val_rng is given. If not, set it to default range
    if val_rng is None:
        val_rng = np.zeros([n_val, 2])
        val_rng[:, 1] = 1

    # Scale all samples to the value range [0, 1]
    sam_set = ((sam_set-val_rng[:, 0])/(val_rng[:, 1]-val_rng[:, 0]))

    # Create empty array of valid samples
    ext_sam_set = []

    # Create lower and upper limits of the hypercube containing samples that
    # can influence the created hypercube
    lower_lim = 0-np.sqrt(n_val)
    upper_lim = 1+np.sqrt(n_val)

    # Check which samples are within val_rng or just outside of it
    for i in range(n_sam):
        # If a sample is within the outer hypercube, save it
        if(((lower_lim <= sam_set[i, :])*(sam_set[i, :] <= upper_lim)).all()):
            ext_sam_set.append(sam_set[i, :])

    # Return sam_set
    return(np.array(ext_sam_set))
