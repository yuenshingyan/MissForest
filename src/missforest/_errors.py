"""This module contains all custom exceptions or errors."""

__all__ = ["MultipleDataTypesError", "NotFittedError"]


class MultipleDataTypesError(Exception):
    """Raised when any of the columns the input argument 'x' has more one
    datatype when calling function '_validate_single_datatype_features'."""

    pass


class NotFittedError(Exception):
    """Raised when class method 'transform' is being called before
    'MissForest' is trained."""

    pass
