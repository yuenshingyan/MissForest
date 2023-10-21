"""This module contains Python class 'MissForest'."""

__all__ = ["InputInvalidError", "MultipleDataTypesError"]
__version__ = "2.0.0"
__author__ = "Yuen Shing Yan Hindy"


class InputInvalidError(Exception):
    """'InputInvalidError' is raised when the input argument 'X' is not
    either pandas dataframe, numpy array or list of lists when calling class
    method '_check_if_valid'."""
    pass


class MultipleDataTypesError(Exception):
    """'MultipleDataTypesError' is raised when any of the columns the input
    argument 'X' has more one datatype when calling class method
    '_check_if_all_single_type'."""
    pass
