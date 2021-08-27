# -*- coding: utf-8 -*-

"""
Utilities
=========
Provides several useful utility functions.

"""


# %% IMPORTS
# Built-in imports
from ast import literal_eval
from inspect import currentframe, getouterframes, isclass, isfunction, ismethod
import logging
import logging.config
import re
import warnings

# e13Tools imports
from e13tools.core import InputError

# All declaration
__all__ = ['add_to_all', 'aux_char_set', 'check_instance', 'delist',
           'docstring_append', 'docstring_copy', 'docstring_substitute',
           'get_main_desc', 'get_outer_frame', 'raise_error', 'raise_warning',
           'split_seq', 'unpack_str_seq']


# %% DECORATOR DEFINITIONS
# Define custom decorator for automatically appending names to __all__
def add_to_all(obj):
    """
    Custom decorator that allows for the name of the provided object `obj` to
    be automatically added to the `__all__` attribute of the frame this
    decorator is used in. The provided `obj` must have a `__name__` attribute.

    """

    # Obtain caller's frame
    frame = currentframe().f_back

    # Get __all__ list in caller's frame
    __all__ = frame.f_globals.get('__all__')

    # If __all__ does not exist yet, make a new one
    if __all__ is None:
        __all__ = []
        frame.f_globals['__all__'] = __all__

    # Append name of given obj to __all__
    if hasattr(obj, '__name__'):
        __all__.append(obj.__name__)
    else:
        raise AttributeError("Input argument 'obj' does not have attribute"
                             "'__name__'!")

    # Return obj
    return(obj)


# Define custom decorator for appending docstrings to a function's docstring
def docstring_append(addendum, join=''):
    """
    Custom decorator that allows a given string `addendum` to be appended to
    the docstring of the target function/class, separated by a given string
    `join`.

    If `addendum` is not a string, its :attr:`~object.__doc__` attribute is
    used instead.

    """

    # If addendum is not a string , try to use its __doc__ attribute
    if not isinstance(addendum, str):
        addendum = addendum.__doc__

    # This function performs the docstring append on a given definition
    def do_append(target):
        # Perform append
        if target.__doc__:
            target.__doc__ = join.join([target.__doc__, addendum])
        else:
            target.__doc__ = addendum

        # Return the target definition
        return(target)

    # Return decorator function
    return(do_append)


# Define custom decorator for copying docstrings from one function to another
def docstring_copy(source):
    """
    Custom decorator that allows the docstring of a function/class `source` to
    be copied to the target function/class.

    """

    # This function performs the docstring copy on a given definition
    def do_copy(target):
        # Check if source has a docstring
        if source.__doc__:
            # Perform copy
            target.__doc__ = source.__doc__

        # Return the target definition
        return(target)

    # Return decorator function
    return(do_copy)


# Define custom decorator for substituting strings into a function's docstring
def docstring_substitute(*args, **kwargs):
    """
    Custom decorator that allows either given positional arguments `args` or
    keyword arguments `kwargs` to be substituted into the docstring of the
    target function/class.

    Both `%` and `.format()` string formatting styles are supported. Keep in
    mind that this decorator will always attempt to do %-formatting first, and
    only uses `.format()` if the first fails.

    """

    # Check if solely args or kwargs were provided
    if len(args) and len(kwargs):
        raise InputError("Either only positional or keyword arguments are "
                         "allowed!")
    else:
        params = args or kwargs

    # This function performs the docstring substitution on a given definition
    def do_substitution(target):
        # Check if target has a docstring that can be substituted to
        if target.__doc__:
            # Make a copy of the target docstring to check formatting later
            doc_presub = str(target.__doc__)

            # Try to use %-formatting
            try:
                target.__doc__ = target.__doc__ % (params)
            # If that raises an error, use .format with *args
            except TypeError:
                target.__doc__ = target.__doc__.format(*params)
            # Using **kwargs with % raises no errors if .format is required
            else:
                # Check if formatting was done and use .format if not
                if(target.__doc__ == doc_presub):
                    target.__doc__ = target.__doc__.format(**params)

        # Raise error if target has no docstring
        else:
            raise InputError("Target has no docstring available for "
                             "substitutions!")

        # Return the target definition
        return(target)

    # Return decorator function
    return(do_substitution)


# %% FUNCTION DEFINITIONS
# This function checks if a given instance was initialized properly
def check_instance(instance, cls):
    """
    Checks if provided `instance` has been initialized from a proper `cls`
    (sub)class. Raises a :class:`~TypeError` if `instance` is not an instance
    of `cls`.

    Parameters
    ----------
    instance : object
        Class instance that needs to be checked.
    cls : class
        The class which `instance` needs to be properly initialized from.

    Returns
    -------
    result : bool
        Bool indicating whether or not the provided `instance` was initialized
        from a proper `cls` (sub)class.

    """

    # Check if cls is a class
    if not isclass(cls):
        raise InputError("Input argument 'cls' must be a class!")

    # Check if instance was initialized from a cls (sub)class
    if not isinstance(instance, cls):
        raise TypeError("Input argument 'instance' must be an instance of the "
                        "%s.%s class!" % (cls.__module__, cls.__name__))

    # Retrieve a list of all cls attributes
    class_attrs = dir(cls)

    # Check if all cls attributes can be called in instance
    for attr in class_attrs:
        if not hasattr(instance, attr):
            return(False)
    else:
        return(True)


# Function that returns a copy of a list with all empty lists/tuples removed
def delist(list_obj):
    """
    Returns a copy of `list_obj` with all empty lists and tuples removed.

    Parameters
    ----------
    list_obj : list
        A list object that requires its empty list/tuple elements to be
        removed.

    Returns
    -------
    delisted_copy : list
        Copy of `list_obj` with all empty lists/tuples removed.

    """

    # Check if list_obj is a list
    if(type(list_obj) != list):
        raise TypeError("Input argument 'list_obj' is not of type 'list'!")

    # Make a copy of itself
    delisted_copy = list(list_obj)

    # Remove all empty lists/tuples from this copy
    off_dex = len(delisted_copy)-1
    for i, element in enumerate(reversed(delisted_copy)):
        # Remove empty lists
        if(isinstance(element, list) and element == []):
            delisted_copy.pop(off_dex-i)
        # Remove empty tuples
        elif(isinstance(element, tuple) and element == ()):
            delisted_copy.pop(off_dex-i)

    # Return the copy
    return(delisted_copy)


# This function retrieves the main description of an object
def get_main_desc(source):
    """
    Retrieves the main description of the provided object `source` and returns
    it.

    The main description is defined as the first paragraph of its docstring.

    Parameters
    ----------
    source : object
        The object whose main description must be retrieved.

    Returns
    -------
    main_desc : str or None
        The main description string of the provided `source` or *None* if
        `source` has not docstring.

    """

    # Retrieve the docstring of provided source
    doc = source.__doc__

    # If doc is None, return None
    if doc is None:
        return(None)

    # Obtain the index of the last character of the first paragraph
    index = doc.find('\n\n')

    # If index is -1, there is only 1 paragraph
    if(index == -1):
        index = len(doc)

    # Gather everything up to this index
    doc = doc[:index]

    # Replace all occurances of 2 or more whitespace characters by a space
    doc = re.sub(r"\s{2,}", ' ', doc)

    # Return doc
    return(doc.strip())


# This function retrieves a specified outer frame of a function
def get_outer_frame(func):
    """
    Checks whether or not the calling function contains an outer frame
    corresponding to `func` and returns it if so. If this frame cannot be
    found, returns *None* instead.

    Parameters
    ----------
    func : function
        The function or method whose frame must be located in the outer frames.

    Returns
    -------
    outer_frame : frame or None
        The requested outer frame if it was found, or *None* if it was not.

    """

    # If func is a function, obtain its name and module name
    if isfunction(func):
        name = func.__name__
        module_name = func.__module__
    # Else, if func is a method, obtain its name and class object
    elif ismethod(func):
        name = func.__name__
        class_obj = func.__self__.__class__
    # Else, raise error
    else:
        raise InputError("Input argument 'func' must be a callable function or"
                         " method!")

    # Obtain the caller's frame
    caller_frame = currentframe().f_back

    # Loop over all outer frames
    for frame_info in getouterframes(caller_frame):
        # Check if frame has the correct name
        if(frame_info.function == name):
            # If func is a function, return if module name is also correct
            if(isfunction(func) and
               frame_info.frame.f_globals['__name__'] == module_name):
                return(frame_info.frame)

            # Else, return frame if class is also correct
            elif(frame_info.frame.f_locals['self'].__class__ is class_obj):
                return(frame_info.frame)
    else:
        return(None)


# This function raises a given error after logging the error
def raise_error(err_msg, err_type=Exception, logger=None, err_traceback=None):
    """
    Raises a given error `err_msg` of type `err_type` and logs the error using
    the provided `logger`.

    Parameters
    ----------
    err_msg : str
        The message included in the error.

    Optional
    --------
    err_type : :class:`Exception` subclass. Default: :class:`Exception`
        The type of error that needs to be raised.
    logger : :obj:`~logging.Logger` object or None. Default: None
        The logger to which the error message must be written.
        If *None*, the :obj:`~logging.RootLogger` logger is used instead.
    err_traceback : traceback object or None. Default: None
        The traceback object that must be used for this exception, useful for
        when this function is used for reraising a caught exception.
        If *None*, no additional traceback is used.

        .. versionadded:: 0.6.17

    See also
    --------
    :func:`~raise_warning`
        Raises and logs a given warning.

    """

    # Log the error
    logger = logging.root if logger is None else logger
    logger.error(err_msg)

    # Create error value
    err_value = err_type(err_msg)

    # Raise error
    if err_value.__traceback__ is not err_traceback:
        raise err_value.with_traceback(err_traceback)
    else:
        raise err_value


# This function raises a given warning after logging the warning
def raise_warning(warn_msg, warn_type=UserWarning, logger=None, stacklevel=1):
    """
    Raises/issues a given warning `warn_msg` of type `warn_type` and logs the
    warning using the provided `logger`.

    Parameters
    ----------
    warn_msg : str
        The message included in the warning.

    Optional
    --------
    warn_type : :class:`Warning` subclass. Default: :class:`UserWarning`
        The type of warning that needs to be raised/issued.
    logger : :obj:`~logging.Logger` object or None. Default: None
        The logger to which the warning message must be written.
        If *None*, the :obj:`~logging.RootLogger` logger is used instead.
    stacklevel : int. Default: 1
        The stack level of the warning message at the location of this function
        call. The actual used stack level is increased by one to account for
        this function call.

    See also
    --------
    :func:`~raise_error`
        Raises and logs a given error.

    """

    # Log the warning and raise it right after
    logger = logging.root if logger is None else logger
    logger.warning(warn_msg)
    warnings.warn(warn_msg, warn_type, stacklevel=stacklevel+1)


# Function for splitting a string or sequence into a list of elements
def split_seq(*seq):
    """
    Converts a provided sequence `seq` to a string, removes all auxiliary
    characters from it, splits it up into individual elements and converts all
    elements back to booleans; floats; integers; and/or strings.

    The auxiliary characters are given by :obj:`~aux_char_set`. One can add,
    change and remove characters from the set if required. If one wishes to
    keep an auxiliary character that is in `seq`, it must be escaped by a
    backslash (note that backslashes themselves also need to be escaped).

    This function can be used to easily unpack a large sequence of nested
    iterables into a single list, or to convert a formatted string to a list of
    elements.

    Parameters
    ----------
    seq : str, array_like or tuple of arguments
        The sequence that needs to be split into individual elements.
        If array_like, `seq` is first unpacked into a string.
        It is possible for `seq` to be a nested iterable.

    Returns
    -------
    new_seq : list
        A list with all individual elements converted to booleans; floats;
        integers; and/or strings.

    Examples
    --------
    The following function calls all produce the same output:

        >>> split_seq('A', 1, 20.0, 'B')
        ['A', 1, 20.0, 'B']
        >>> split_seq(['A', 1, 2e1, 'B'])
        ['A', 1, 20.0, 'B']
        >>> split_seq("A 1 20. B")
        ['A', 1, 20.0, 'B']
        >>> split_seq([("A", 1), (["20."], "B")])
        ['A', 1, 20.0, 'B']
        >>> split_seq("[(A / }| ; <1{}) , ,>20.0000 !! < )?% \\B")
        ['A', 1, 20.0, 'B']

    If one wants to keep the '?' in the last string above, it must be escaped:

        >>> split_seq("[(A / }| ; <1{}) , ,>20.0000 !! < )\\?% \\B")
        ['A', 1, 20.0, '?', 'B']

    See also
    --------
    :func:`~unpack_str_seq`
        Unpacks a provided (nested) sequence into a single string.

    """

    # Unpack the provided sequence into a list of characters
    seq = list(unpack_str_seq(*seq, sep='\n'))

    # Process all backslashes
    for index, char in enumerate(seq):
        # If char is a backslash
        if(char == '\\'):
            # If this backslash is escaped, skip
            if(index != 0 and seq[index-1] is None):
                pass
            # Else, if this backslash escapes a character, replace by None
            elif(index != len(seq)-1 and seq[index+1] in aux_char_set):
                seq[index] = None

    # Remove all unwanted characters from the string, except those escaped
    for char in aux_char_set:
        # Set the search index
        index = 0

        # Keep looking for the specified character
        while True:
            # Check if the character can be found in seq or break if not
            try:
                index = seq.index(char, index)
            except ValueError:
                break

            # If so, remove it if it was not escaped
            if(index == 0 or seq[index-1] is not None):
                seq[index] = '\n'
            # If it was escaped, remove None instead
            else:
                seq[index-1] = ''

            # Increment search index by 1
            index += 1

    # Convert seq back to a single string
    seq = ''.join(seq)

    # Split sequence up into elements
    seq = seq.split('\n')

    # Remove all empty strings
    while '' in seq:
        seq.remove('')

    # Loop over all elements in seq
    for i, val in enumerate(seq):
        # Try to convert back to bool/float/int using literal_eval
        try:
            seq[i] = literal_eval(val)
        # If it cannot be evaluated using literal_eval, save as string
        except (ValueError, SyntaxError):
            seq[i] = val

    # Return it
    return(seq)


# List/set of auxiliary characters to be used in split_seq()
aux_char_set = set(['(', ')', '[', ']', ',', "'", '"', '|', '/', '\\', '{',
                    '}', '<', '>', '´', '¨', '`', '?', '!', '%', ':', ';', '=',
                    '$', '~', '#', '@', '^', '&', '*', '“', '’', '”', '‘',
                    ' ', '\t'])


# Function that unpacks a provided sequence of iterables to a single string
def unpack_str_seq(*seq, sep=', '):
    """
    Unpacks a provided sequence `seq` of elements and iterables, and converts
    it to a single string separated by `sep`.

    Use :func:`~split_seq` if it is instead required to unpack `seq` into a
    single list while maintaining the types of all elements.

    Parameters
    ----------
    seq : str, array_like or tuple of arguments
        The sequence that needs to be unpacked into a single string.
        If `seq` contains nested iterables, this function is used recursively
        to unpack them as well.

    Optional
    --------
    sep : str. Default: ', '
        The string to use for separating the elements in the unpacked string.

    Returns
    -------
    unpacked_seq : str
        A string containing all elements in `seq` unpacked and converted to a
        string, separated by `sep`.

    Examples
    --------
    The following function calls all produce the same output:

        >>> unpack_str_seq('A', 1, 20.0, 'B')
        'A, 1, 20.0, B'
        >>> unpack_str_seq(['A', 1, 2e1, 'B'])
        'A, 1, 20.0, B'
        >>> unpack_str_seq("A, 1, 20.0, B")
        'A, 1, 20.0, B'
        >>> unpack_str_seq([("A", 1), (["20.0"], "B")])
        'A, 1, 20.0, B'

    See also
    --------
    :func:`~split_seq`
        Splits up a provided (nested) sequence into a list of individual
        elements.

    """

    # Check if provided separator is a string
    if not isinstance(sep, str):
        raise TypeError("Input argument 'sep' is not of type 'str'!")

    # Convert provided sequence to a list
    seq = list(seq)

    # Loop over all elements in seq and unpack iterables to strings as well
    for i, arg in enumerate(seq):
        # If arg can be iterated over and is not a string, unpack it
        if isinstance(arg, (list, tuple, set)):
            seq[i] = unpack_str_seq(*arg, sep=sep)

    # Join entire sequence together to a single string
    unpacked_seq = sep.join(map(str, seq))

    # Return unpacked_seq
    return(unpacked_seq)
