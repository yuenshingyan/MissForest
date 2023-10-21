"""This module contains Python class 'MissForest'."""

__all__ = ["InputInvalidError"]
__version__ = "2.0.0"
__author__ = "Yuen Shing Yan Hindy"


class InputInvalidError(Exception):
    """'InputInvalidError' is raised when the input argument 'X' is not
    either pandas dataframe, numpy array or list of lists."""
    pass
