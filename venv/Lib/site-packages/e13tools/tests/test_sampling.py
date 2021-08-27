# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import numpy as np
import pytest

# e13Tools imports
from e13tools.core import ShapeError
from e13tools.sampling import lhd


# %% PYTEST HELPER FUNCTIONS
# Create fixture that resets the NumPy random seed for every test
@pytest.fixture()
def set_numpy_random_seed():
    np.random.seed(0)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest class for lhd()-function
@pytest.mark.usefixtures('set_numpy_random_seed')
class Test_lhd(object):
    # Test 3-2 LHD with random samples
    def test_3_2_random(self):
        assert np.allclose(lhd(3, 2, method='random'),
                           np.array([[0.18293783, 0.47919574],
                                     [0.86758779, 0.96392433],
                                     [0.57172979, 0.21529804]]))

    # Test 3-2 LHD with fixed samples
    def test_3_2_fixed(self):
        assert np.allclose(lhd(3, 2, method='fixed'),
                           np.array([[1, 1], [0.5, 0], [0, 0.5]]))

    # Test 3-2 LHD with centered samples
    def test_3_2_center(self):
        assert np.allclose(lhd(3, 2, method='center'),
                           np.array([[0.83333333, 0.83333333],
                                     [0.5, 0.16666667], [0.16666667, 0.5]]))

    # Test 3-2 LHD with specified value range
    def test_3_2_custom_rng(self):
        val_rng = [[0, 2], [1, 4]]
        assert np.allclose(lhd(3, 2, val_rng=val_rng),
                           np.array([[0.36587567, 2.43758721],
                                     [1.73517558, 3.89177300],
                                     [1.14345958, 1.64589411]]))

    # Test 3-3 LHD with invalid val_rng shape raises an error
    def test_3_3_invalid_val_rng_shape(self):
        val_rng = [[0, 2], [0, 1]]
        with pytest.raises(ShapeError):
            lhd(3, 3, val_rng=val_rng)

    # Test 3-2 LHD with constraints given
    def test_3_2_constraints(self):
        constraints = np.random.rand(2, 2)
        assert np.allclose(lhd(3, 2, constraints=constraints, criterion=0),
                           np.array([[1., 1], [0.5, 0], [0, 0.5]]))

    # Test 3-2 LHD with constraints given same sample as LHD
    def test_3_2_same_constraints(self):
        constraints = [0, 0]
        assert np.allclose(lhd(3, 2, constraints=constraints, criterion=0),
                           np.array([[0, 1], [1, 0], [0.5, 0.5]]))

    # Test 3-2 LHD with constraints given outside of val_rng
    def test_3_2_constraints_outside(self):
        constraints = np.random.rand(2, 2)
        val_rng = [[1, 2], [2, 3]]
        assert np.allclose(lhd(3, 2, val_rng=val_rng,
                               constraints=constraints),
                           np.array([[1.54863137, 2.46114717],
                                     [1.14121827, 2.32122092],
                                     [1.81252907, 2.93057501]]))

    # Test 3-2 LHD with no constraints given
    def test_3_2_no_constraints(self):
        assert np.allclose(lhd(3, 2, constraints=[]),
                           np.array([[0.18293783, 0.47919574],
                                     [0.86758779, 0.96392433],
                                     [0.57172979, 0.21529804]]))

    # Test 3-2 LHD with 3D constraints raises an error
    def test_3_2_3D_constraints(self):
        constraints = np.random.rand(2, 2, 2)
        with pytest.raises(ShapeError):
            lhd(3, 2, constraints=constraints)

    # Test 3-2 LHD with invalid constraints raises an error
    def test_3_2_invalid_constraints(self):
        constraints = np.random.rand(2, 3)
        with pytest.raises(ShapeError):
            lhd(3, 2, constraints=constraints)

    # Test 3-2 LHD with maximin criterion, string
    def test_3_2_maximin_str(self):
        assert np.allclose(lhd(3, 2, method='random', criterion='maximin'),
                           np.array([[1, 1], [0.5, 0], [0, 0.5]]))

    # Test 3-2 LHD with maximin criterion, float
    def test_3_2_maximin_float(self):
        assert np.allclose(lhd(3, 2, method='fixed', criterion=0),
                           np.array([[1, 1], [0.5, 0], [0, 0.5]]))

    # Test 3-2 LHD with correlation criterion, string
    def test_3_2_correlation_str(self):
        assert np.allclose(lhd(3, 2, method='center', criterion='correlation'),
                           np.array([[0.83333333, 0.83333333],
                                     [0.5, 0.16666667], [0.16666667, 0.5]]))

    # Test 3-2 LHD with correlation criterion, float
    def test_3_2_correlation_float(self):
        assert np.allclose(lhd(3, 2, method='center', criterion=1),
                           np.array([[0.83333333, 0.83333333],
                                     [0.5, 0.16666667], [0.16666667, 0.5]]))

    # Test 3-2 LHD with multi criterion, string
    def test_3_2_multi_str(self):
        assert np.allclose(lhd(3, 2, method='center', criterion='multi'),
                           np.array([[0.83333333, 0.83333333],
                                     [0.5, 0.16666667], [0.16666667, 0.5]]))

    # Test 3-2 LHD with multi criterion, float
    def test_3_2_multi_float(self):
        assert np.allclose(lhd(3, 2, method='fixed', criterion=0.5),
                           np.array([[1, 1], [0.5, 0], [0, 0.5]]))

    # Test 1-2 LHD with fixed samples
    def test_1_2_fixed(self):
        assert np.allclose(lhd(1, 2, method='fixed', criterion=0),
                           np.array([[0.5, 0.5]]))

    # Test 2-2 LHD with fixed samples
    def test_2_2_fixed(self):
        assert np.allclose(lhd(2, 2, method='fixed', criterion=0),
                           np.array([[1, 0], [0, 1]]))

    # Test 3-1 LHD with fixed samples
    def test_3_1_fixed(self):
        assert np.allclose(lhd(3, 1, method='fixed', criterion=0),
                           np.array([[1], [0.5], [0]]))

    # Test 3-2 LHD with invalid criterion raises an error, string
    def test_3_2_invalid_criterion_str(self):
        with pytest.raises(ValueError):
            lhd(3, 2, criterion='test')

    # Test 3-2 LHD with invalid criterion raises an error, float
    def test_3_2_invalid_criterion_float(self):
        with pytest.raises(ValueError):
            lhd(3, 2, criterion=1.5)

    # Test 3-2 LHD and request the score
    def test_3_2_score(self):
        results = lhd(3, 2, criterion=0, get_score=True)
        assert np.allclose(results[0], np.array([[1, 1], [0.5, 0], [0, 0.5]]))
        assert np.allclose(results[1], np.array([0.80444958, 0.5, 0.80444958]))

    # Test 4-2 LHD and do not request quick-scan
    def test_4_2_no_quick_scan(self):
        assert np.allclose(lhd(4, 2, criterion=0, iterations=100,
                               quickscan=False),
                           np.array([[0.33333333, 0], [0, 0.66666667],
                                     [1, 0.33333333], [0.66666667, 1]]))
