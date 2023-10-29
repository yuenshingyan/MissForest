"""This module contains Python class 'MissForest'."""

__all__ = ["MultipleDataTypesError", "NotFittedError"]
__version__ = "2.0.0"
__author__ = "Yuen Shing Yan Hindy"


class MultipleDataTypesError(Exception):
    """'MultipleDataTypesError' is raised when any of the columns the input
    argument 'X' has more one datatype when calling class method
    '_check_if_all_single_type'."""
    pass


class NotFittedError(Exception):
    """'NotTrainedError is raised when class method transform is being called
    before 'MissForest' is trained.'"""
    pass
