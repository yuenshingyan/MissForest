"""This module contains all custom exceptions or errors."""

__all__ = ["MultipleDataTypesError", "NotFittedError"]


class MultipleDataTypesError(Exception):
    """Raised when any column of the input argument `x` has more than one
    datatype when calling the function `_validate_single_datatype_features`.
    """
    pass


class NotFittedError(Exception):
    """Exception raised when attempting to call the class method `transform`
    before the `MissForest` model has been trained."""
    pass
