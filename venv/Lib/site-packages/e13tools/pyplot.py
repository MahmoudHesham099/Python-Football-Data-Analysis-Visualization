# -*- coding: utf-8 -*-

"""
PyPlot
======
Provides a collection of functions useful in various plotting routines.

"""


# %% IMPORTS
# Package imports
try:
    import astropy.units as apu
    import_astropy = 1
except ImportError:  # pragma: no cover
    import_astropy = 0
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

# e13Tools imports
from e13tools.core import InputError

# All declaration
__all__ = ['apu2tex', 'center_spines', 'draw_textline', 'f2tex', 'q2tex']


# %% FUNCTIONS
# This function converts an astropy unit into a TeX string
def apu2tex(unit, unitfrac=False):
    """
    Transform a :obj:`~astropy.units.Unit` object into a (La)TeX string for
    usage in a :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    unit : :obj:`~astropy.units.Unit` object
        Unit to be transformed.

    Optional
    --------
    unitfrac : bool. Default: False
        Whether or not to write `unit` as a LaTeX fraction.

    Returns
    -------
    out : string
        String containing `unit` written in (La)TeX string.

    Examples
    --------
    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass)
    '\\mathrm{M_{\\odot}}'

    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass/apu.yr, unitfrac=False)
    '\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> import astropy.units as apu
    >>> apu2tex(apu.solMass/apu.yr, unitfrac=True)
    '\\mathrm{\\frac{M_{\\odot}}{yr}}'

    """

    if import_astropy:
        if not unitfrac:
            string = unit.to_string('latex_inline')
        else:
            string = unit.to_string('latex')

        # Remove '$' from the string and make output a string (py2.7)
        return(str(string.replace("$", "")))
    else:  # pragma: no cover
        raise ImportError("This function requires AstroPy!")


# This function centers the axes of the provided axes
def center_spines(centerx=0, centery=0, set_xticker=False, set_yticker=False,
                  ax=None):
    """
    Centers the axis spines at <`centerx`, `centery`> on the axis `ax` in a
    :obj:`~matplotlib.figure.Figure` instance. Centers the axis spines at the
    origin by default.

    Optional
    --------
    centerx : int or float. Default: 0
        Centers x-axis at value `centerx`.
    centery : int or float. Default: 0
        Centers y-axis at value `centery`.
    set_xticker : int, float or False. Default: False
        If int or float, sets the x-axis ticker to `set_xticker`.
        If *False*, let :obj:`~matplotlib.figure.Figure` instance decide.
    set_yticker : int, float or False. Default: False
        If int or float, sets the y-axis ticker to `set_yticker`.
        If *False*, let :obj:`~matplotlib.figure.Figure` instance decide.
    ax : :obj:`~matplotlib.axes.Axes` object or None. Default: None
        If :obj:`~matplotlib.axes.Axes` object, centers the axis spines
        of specified :obj:`~matplotlib.figure.Figure` instance.
        If *None*, centers the axis spines of current
        :obj:`~matplotlib.figure.Figure` instance.

    """

    # If no AxesSubplot object is provided, make one
    if ax is None:
        ax = plt.gca()

    # Set the axis's spines to be centered at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)
    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_position(('data', centerx))
    ax.spines['top'].set_position(('data', centery))

    # Hide the line (but not ticks) for "extra" spines
    for side in ['right', 'top']:
        ax.spines[side].set_color('none')

    # On both the x and y axes...
    for axis, center in zip([ax.xaxis, ax.yaxis], [centerx, centery]):
        # TODO: STILL HAVE TO FIX THAT THE TICKLABELS ARE ALWAYS HIDDEN
        # Hide the ticklabels at <centerx, centery>
        formatter = mpl.ticker.ScalarFormatter()
        formatter.center = center
        axis.set_major_formatter(formatter)

    # Add origin offset ticklabel if <centerx=0, centery=0> using annotation
    if(centerx == 0 and centery == 0):
        xlabel, ylabel = map(formatter.format_data, [centerx, centery])
        ax.annotate("0", (centerx, centery), xytext=(-4, -4),
                    textcoords='offset points', ha='right', va='top')

    # Set x-axis ticker
    if set_xticker:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(set_xticker))

    # Set y-axis ticker
    if set_yticker:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(set_yticker))


# This function draws a line with text in the provided figure
def draw_textline(text, *, x=None, y=None, pos='start top', ax=None,
                  line_kwargs={}, text_kwargs={}):
    """
    Draws a line on the axis `ax` in a :obj:`~matplotlib.figure.Figure`
    instance and prints `text` on top.

    Parameters
    ----------
    text : str
        Text to be printed on the line.
    x : scalar or None
        If scalar, text/line x-coordinate.
        If *None*, line covers complete x-axis.
        Either `x` or `y` needs to be *None*.
    y : scalar or None
        If scalar, text/line y-coordinate.
        If *None*, line covers complete y-axis.
        Either `x` or `y` needs to be *None*.

    Optional
    --------
    pos : {'start', 'end'}{'top', 'bottom'}. Default: 'start top'
        If 'start', prints the text at the start of the drawn line.
        If 'end', prints the text at the end of the drawn line.
        If 'top', prints the text above the drawn line.
        If 'bottom', prints the text below the drawn line.
        Arguments must be given as a single string.
    ax : :obj:`~matplotlib.axes.Axes` object or None. Default: None
        If :obj:`~matplotlib.axes.Axes` object, draws line in specified
        :obj:`~matplotlib.figure.Figure` instance.
        If *None*, draws line in current :obj:`~matplotlib.figure.Figure`
        instance.
    line_kwargs : dict of :class:`~matplotlib.lines.Line2D` properties.\
        Default: {}
        The keyword arguments used for drawing the line.
    text_kwargs : dict of :class:`~matplotlib.text.Text` properties.\
        Default: {}
        The keyword arguments used for drawing the text.

    """

    # If no AxesSubplot object is provided, make one
    if ax is None:
        ax = plt.gca()

    # Convert pos to lowercase
    pos = pos.lower()

    # Set default line_kwargs and text_kwargs
    default_line_kwargs = {'linestyle': '-',
                           'color': 'k'}
    default_text_kwargs = {'color': 'k',
                           'fontsize': 14}

    # Combine given kwargs with default ones
    default_line_kwargs.update(line_kwargs)
    default_text_kwargs.update(text_kwargs)
    line_kwargs = default_line_kwargs
    text_kwargs = default_text_kwargs

    # Check if certain keyword arguments are present in text_fmt
    text_keys = list(text_kwargs.keys())
    for key in text_keys:
        if key in ('va', 'ha', 'verticalalignment', 'horizontalalignment',
                   'rotation', 'transform', 'x', 'y', 's'):
            text_kwargs.pop(key)

    # Set line specific variables
    if x is None and y is not None:
        ax.axhline(y, **line_kwargs)
    elif x is not None and y is None:
        ax.axvline(x, **line_kwargs)
    else:
        raise InputError("Either of input arguments 'x' and 'y' needs to be "
                         "*None*!")

    # Gather case specific text properties
    if ('start') in pos and ('top') in pos:
        ha = 'left' if x is None else 'right'
        va = 'bottom'
        other_axis = 0
    elif ('start') in pos and ('bottom') in pos:
        ha = 'left'
        va = 'top' if x is None else 'bottom'
        other_axis = 0
    elif ('end') in pos and ('top') in pos:
        ha = 'right'
        va = 'bottom' if x is None else 'top'
        other_axis = 1
    elif ('end') in pos and ('bottom') in pos:
        ha = 'right' if x is None else 'left'
        va = 'top'
        other_axis = 1
    else:
        raise ValueError("Input argument 'pos' is invalid!")

    # Set proper axes and rotation
    if x is None:
        x = other_axis
        rotation = 0
        transform = transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
    else:
        y = other_axis
        rotation = 90
        transform = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)

    # Draw text
    ax.text(x, y, text, rotation=rotation, ha=ha, va=va,
            transform=transform, **text_kwargs)


# This function converts a float into a TeX string
def f2tex(value, *errs, sdigits=4, power=3, nobase1=True):
    """
    Transform a value into a (La)TeX string for usage in a
    :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    value : int or float
        Value to be transformed.

    Optional
    --------
    errs : int or float
        The upper and lower :math:`1\\sigma`-errors of the given `value`.
        If only a single value is given, `value` is assumed to have a centered
        error interval of `errs`.
    sdigits : int. Default: 4
        Number of significant digits any value is returned with.
    power : int. Default: 3
        Minimum abs(log10(`value`)) required before all values are written in
        scientific form.
        This value is ignored if `sdigits` forces scientific form to (not) be
        used.
    nobase1 : bool. Default: True
        Whether or not to include `base` in scientific form if `base=1`.
        This is always *False* if `errs` contains at least one value.

    Returns
    -------
    out : string
        String containing `value` and `errs` written in (La)TeX string.

    Examples
    --------
    >>> f2tex(20.2935826592)
    '20.29'

    >>> f2tex(20.2935826592, sdigits=6)
    '20.2936'

    >>> f2tex(20.2935826592, power=1)
    '2.029\\cdot 10^{1}'

    >>> f2tex(1e6, nobase1=True)
    '10^{6}'

    >>> f2tex(1e6, nobase1=False)
    '1.000\\cdot 10^{6}'

    >>> f2tex(20.2935826592, 0.1)
    '20.29\\pm 0.10'

    >>> f2tex(20.2935826592, 0.1, 0.2)
    '20.29^{+0.10}_{-0.20}'

    >>> f2tex(1e6, 12, 10)
    '1.000^{+0.000}_{-0.000}\\cdot 10^{6}'

    >>> f2tex(1e6, 12, 10, sdigits=6)
    '1.000^{+0.000}_{-0.000}\\cdot 10^{6}'

    """

    # Collect value and errs together
    vals = [value, *map(abs, errs)]

    # If vals contains more than 1 value, set nobase1 to False
    if(len(vals) > 1):
        nobase1 = False

    # Calculate the maximum power required for all values
    n = [int(np.floor(np.log10(abs(v)))) if v else -np.infty for v in vals]
    n_max = max(n)

    # Check that n_max is a valid value
    if(n_max == -np.infty):
        sdigits = 0
        n_max = 0

    # If there are no significant digits requested, never use scientific form
    if not sdigits:
        power = None
    # Else, if n_max >= sdigits, always use scientific form
    elif(n_max >= sdigits):
        power = 0

    # Create empty list of string representations
    strings = []

    # Convert all values into their proper string representations
    for v, ni in zip(vals, n):
        # Calculate the number of significant digits each value should have
        sd = sdigits-(n_max-ni)

        # If the sd is zero or -infinity
        if(sd <= 0):
            # Then v must always be zero
            v *= 0.
            sd = max(0, sdigits-n_max)

        # If no power is required, create string without scientific form
        if power is None or (abs(n_max) < power):
            strings.append(r"{0:#.{1}g}".format(v, sd))
            pow_str = ""

        # Else, convert value to scientific form
        else:
            # If v is zero, set sd to the maximum number of significant digits
            if not v:
                sd = sdigits

            # Calculate the base value
            base = v/pow(10, n_max)

            # Determine string representation
            if(base == 1) and nobase1:
                strings.append(r"10^{{{0}}}".format(n_max))
                pow_str = ""
            else:
                strings.append(r"{0:#.{1}g}".format(base, sd))
                pow_str = r"\cdot 10^{{{0}}}".format(n_max)

    # Check contents of strings and convert accordingly
    if(len(strings) == 1):
        fmt = r"{0}{1}"
    elif(len(strings) == 2):
        fmt = r"{0}\pm {1}{2}"
    else:
        fmt = r"{0}^{{+{1}}}_{{-{2}}}{3}"

    # Return string
    return(fmt.format(*strings, pow_str))


# This function converts an astropy quantity into a TeX string
def q2tex(quantity, *errs, sdigits=4, power=3, nobase1=True, unitfrac=False):
    """
    Combination of :func:`~e13tools.pyplot.f2tex` and
    :func:`~e13tools.pyplot.apu2tex`.

    Transform a :obj:`~astropy.units.quantity.Quantity` object into a (La)TeX
    string for usage in a :obj:`~matplotlib.figure.Figure` instance.

    Parameters
    ----------
    quantity : int, float or :obj:`~astropy.units.quantity.Quantity` object
        Quantity to be transformed.

    Optional
    --------
    errs : int, float or :obj:`~astropy.units.quantity.Quantity` object
        The upper and lower :math:`1\\sigma`-errors of the given `quantity`.
        If only a single value is given, `quantity` is assumed to have a
        centered error interval of `errs`.
        The unit of `errs` must be convertible to the unit of `quantity`.
    sdigits : int. Default: 4
        Maximum amount of significant digits any quantity is returned with.
    power : int. Default: 3
        Minimum abs(log10(`value`)) required before all quantities are written
        in scientific form.
        This value is ignored if `sdigits` forces scientific form to (not) be
        used.
    nobase1 : bool. Default: True
        Whether or not to include `base` in scientific form if `base=1`.
        This is always *False* if `errs` contains a value.
    unitfrac : bool. Default: False
        Whether or not to write `unit` as a LaTeX fraction.

    Returns
    -------
    out : string
        String containing `quantity` and `errs` written in (La)TeX string.

    Examples
    --------
    >>> import astropy.units as apu
    >>> q2tex(20.2935826592)
    '20.29'

    >>> q2tex(20.2935826592*apu.kg, 1500*apu.g)
    '20.29\\pm 1.50\\,\\mathrm{kg}'

    >>> q2tex(20.2935826592*apu.solMass/apu.yr)
    '20.29\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> q2tex(20.2935826592*apu.solMass/apu.yr, sdigits=6)
    '20.2936\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> q2tex(20.2935826592*apu.solMass/apu.yr, power=1)
    '2.029\\cdot 10^{1}\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> q2tex(1e6*apu.solMass/apu.yr, nobase1=True)
    '10^{6}\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> q2tex(1e6*apu.solMass/apu.yr, nobase1=False)
    '1.000\\cdot 10^{6}\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> q2tex(20.2935826592*apu.solMass/apu.yr, unitfrac=False)
    '20.29\\,\\mathrm{M_{\\odot}\\,yr^{-1}}'

    >>> q2tex(20.2935826592*apu.solMass, 1*apu.solMass, unitfrac=True)
    '20.29\\pm 1.00\\,\\mathrm{M_{\\odot}}'

    """

    # Collect quantity and errs together
    qnts = [quantity, *errs]

    # If astropy is importable, check if there are quantities
    if import_astropy:
        # Make empty lists of values and units
        values = []
        units = []

        # Loop over all quantities given and split them up into value and unit
        for q in qnts:
            if isinstance(q, apu.quantity.Quantity):
                values.append(q.value)
                units.append(q.unit)
            else:
                values.append(q)
                units.append(apu.dimensionless_unscaled)

        # Obtain the unit of the main value
        unit = units[0]

        # Loop over the errors
        for i, u in enumerate(units[1:], 1):
            # Try to convert the error quantity to have the same unit as main
            try:
                values[i] *= u.to(unit)
            # If this fails, raise error
            except apu.UnitConversionError:
                raise ValueError("Input argument 'errs[{}]' (unit: {!r}; {}) "
                                 "cannot be converted to the same unit as "
                                 "'quantity' (unit: {!r}; {})!".format(
                                     i-1, str(u), u.physical_type,
                                     str(unit), unit.physical_type))

        # Value handling
        string = f2tex(*values, sdigits=sdigits, power=power, nobase1=nobase1)

        # Unit handling
        if(unit.physical_type != 'dimensionless'):
            unit_string = apu2tex(unit, unitfrac=unitfrac)
            string = ''.join([string, r'\,', unit_string])

        # Return string
        return(string)

    # Else, handle given arguments as normal values
    else:  # pragma: no cover
        return(f2tex(*qnts, sdigits=sdigits, power=power, nobase1=nobase1))
