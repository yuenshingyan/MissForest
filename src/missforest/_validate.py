"""This module contains all validation functions."""

from ._info import VERSION, AUTHOR

__all__ = [
    "_is_estimator",
    "_validate_feature_dtype_consistency",
    "_is_numerical_matrix",
    "_is_array_2d",
]
__version__ = VERSION
__author__ = AUTHOR


from typing import Any, Union
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from ._errors import MultipleDataTypesError


def _is_array_2d(array: Any) -> bool:
    """Checks if argument `array` is 2D.

    Parameters
    ----------
    array : Any
        Array to be checked.

    Returns
    -------
    bool
        - True, if argument `array` is 2D.
        - False, if argument `array` is not 2D.
    """
    if len(array.shape) != 2:
        return False

    return True


def _is_numerical_matrix(mat: Any) -> bool:
    """Checks if argument `mat` is fully numerical.

    Parameters
    ----------
    mat : Any
        Matrix to be validated.

    Returns
    -------
    bool
        - True, if `mat` is fully numerical.
        - False, if `mat` is not fully numerical.
    """
    try:
        mat.astype(float)
        return True
    except ValueError:
        return False


def _is_estimator(estimator: Union[Any, BaseEstimator]) -> bool:
    """Checks if the argument `estimator` is an object that implements the
    scikit-learn estimator API.

    Parameters
    ----------
    estimator : estimator object
        This object is assumed to implement the scikit-learn estimator API.

    Returns
    -------
    bool
        - True, if argument `estimator` has class methods `fit` and `predict`.
        - False, if argument `estimator` does not have class methods `fit`
        and `predict`.
        - False, if AttributeError raised.
    """
    try:
        # Check if class methods `fit` and `predict` exist and callable.
        return (
                callable(getattr(estimator, "fit")) and
                callable(getattr(estimator, "predict"))
        )
    except AttributeError:
        return False


def _validate_feature_dtype_consistency(x: pd.DataFrame) -> None:
    """Checks if all values in the features belong to the same datatype.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needs to be checked.

    Raises
    ------
    MultipleDataTypesError
        Raised if not all values in the features belong to the same datatype.
    """
    vectorized_type = np.vectorize(type)
    for c in x.columns:
        all_type = vectorized_type(x[c].dropna())
        if len(pd.unique(all_type)) > 1:
            raise MultipleDataTypesError(
                f"Multiple data types found in feature `{c}`.")
