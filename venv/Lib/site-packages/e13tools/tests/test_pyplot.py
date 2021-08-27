# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
import astropy.units as apu
import matplotlib.pyplot as plt
import pytest

# e13Tools imports
from e13tools.core import InputError
from e13tools.pyplot import (
    apu2tex, center_spines, draw_textline, f2tex, q2tex)

# Save the path to this directory
dirpath = path.dirname(__file__)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for apu2tex()-function
def test_apu2tex():
    assert apu2tex(apu.solMass) == r"\mathrm{M_{\odot}}"
    assert apu2tex(apu.solMass/apu.yr, unitfrac=False) ==\
        r"\mathrm{M_{\odot}\,yr^{-1}}"
    assert apu2tex(apu.solMass/apu.yr, unitfrac=True) ==\
        r"\mathrm{\frac{M_{\odot}}{yr}}"


# Pytest class for center_spines()-function
class Test_center_spines(object):
    # Test if default values work
    def test_default(self):
        fig = plt.figure()
        center_spines()
        plt.close(fig)

    # Test if setting the x and y tickers work
    def test_set_tickers(self):
        fig = plt.figure()
        plt.plot([-1, 1], [-1, 1])
        center_spines(set_xticker=1, set_yticker=1)
        plt.close(fig)


# Pytest class for draw_textline()-function
class Test_draw_textline(object):
    # Test if writing 'test' on the x-axis works, number 1
    def test_x_axis1(self):
        fig = plt.figure()
        draw_textline("test", x=-1, text_kwargs={'va': None})
        plt.close(fig)

    # Test if writing 'test' on the x-axis works, number 2
    def test_x_axis2(self):
        fig = plt.figure()
        draw_textline("test", x=2)
        plt.close(fig)

    # Test if writing 'test' on the y-axis works, number 1
    def test_y_axis1(self):
        fig = plt.figure()
        draw_textline("test", y=-1)
        plt.close(fig)

    # Test if writing 'test' on the y-axis works, number 2
    def test_y_axis2(self):
        fig = plt.figure()
        draw_textline("test", y=2)
        plt.close(fig)

    # Test if writing 'test' on the x-axis works, end-top pos
    def test_x_axis_end_top(self):
        fig = plt.figure()
        draw_textline("test", x=-1, pos="end top")
        plt.close(fig)

    # Test if writing 'test' on the x-axis works, start-bottom pos
    def test_x_axis_start_bottom(self):
        fig = plt.figure()
        draw_textline("test", x=-1, pos="start bottom")
        plt.close(fig)

    # Test if writing 'test' on the x-axis works, end-bottom pos
    def test_x_axis_end_bottom(self):
        fig = plt.figure()
        draw_textline("test", x=-1, pos="end bottom")
        plt.close(fig)

    # Test if writing 'test' on the x-axis y-axis fails
    def test_xy_axis(self):
        fig = plt.figure()
        with pytest.raises(InputError):
            draw_textline("test", x=-1, y=-1)
        plt.close(fig)

    # Test if writing 'test' on the x-axis fails for invalid pos
    def test_x_axis_invalid_pos(self):
        fig = plt.figure()
        with pytest.raises(ValueError):
            draw_textline("test", x=-1, pos="test")
        plt.close(fig)


# Pytest for f2tex()-function
def test_f2tex():
    assert f2tex(20.2935826592) == "20.29"
    assert f2tex(20.2935826592, sdigits=6) == "20.2936"
    assert f2tex(20.2935826592, power=1) == r"2.029\cdot 10^{1}"
    assert f2tex(1e6, nobase1=True) == "10^{6}"
    assert f2tex(1e6, nobase1=False) == r"1.000\cdot 10^{6}"
    assert f2tex(0) == "0."
    assert f2tex(20.2935826592, 0.1) == r"20.29\pm 0.10"
    assert f2tex(20.2935826592, 0.1, 0.2) == "20.29^{+0.10}_{-0.20}"
    assert f2tex(20.2935826592, 0.1, 0.0) == "20.29^{+0.10}_{-0.00}"
    assert f2tex(1e6, 12, 10) == r"1.000^{+0.000}_{-0.000}\cdot 10^{6}"


# Pytest for q2tex()-function
def test_q2tex():
    assert q2tex(20.2935826592) == "20.29"
    assert q2tex(20.2935826592*apu.solMass/apu.yr) ==\
        r"20.29\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, sdigits=6) ==\
        r"20.2936\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, power=1) ==\
        r"2.029\cdot 10^{1}\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(1e6*apu.solMass/apu.yr, nobase1=True) ==\
        r"10^{6}\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(1e6*apu.solMass/apu.yr, nobase1=False) ==\
        r"1.000\cdot 10^{6}\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=False) ==\
        r"20.29\,\mathrm{M_{\odot}\,yr^{-1}}"
    assert q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=True) ==\
        r"20.29\,\mathrm{\frac{M_{\odot}}{yr}}"
    assert q2tex(20.2935826592*apu.kg, 1500*apu.g) ==\
        r"20.29\pm 1.50\,\mathrm{kg}"
    with pytest.raises(ValueError):
        q2tex(1, 1*apu.kg)
