"""This module contains all validation functions."""

from ._info import VERSION, AUTHOR

__all__ = ["_is_estimator", "_validate_single_datatype_features"]
__version__ = VERSION
__author__ = AUTHOR


from typing import Any, Union
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from ._errors import MultipleDataTypesError


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
        Returns True if the argument `estimator` is None or has class
        methods `fit` and `predict`. Otherwise, returns False.
    """
    try:
        # Check if class methods `fit` and `predict` exist and callable.
        is_has_fit_method = callable(getattr(estimator, "fit"))
        is_has_predict_method = callable(getattr(estimator, "predict"))

        return is_has_fit_method and is_has_predict_method
    except AttributeError:
        return False


def _validate_single_datatype_features(x: pd.DataFrame) -> None:
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
                f"Multiple data types found in feature {c}.")
