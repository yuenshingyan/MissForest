"""This module contains validation functions."""

__all__ = [
    "_is_estimator",
    "_validate_single_datatype_features",
]


from typing import Any, Union
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from ._errors import MultipleDataTypesError


def _is_estimator(estimator: Union[Any, BaseEstimator]) -> bool:
    """
    Checks if argument 'estimator' is an object that implement the
    scikit-learn estimator api.

    Parameters
    ----------
    estimator : estimator object
        This object is assumed to implement the scikit-learn estimator api.

    Returns
    -------
    Bool
        If the argument 'estimator' is None or has class method 'fit' and
        'predict', return True. Otherwise, return False.
    """

    try:
        # get the class methods 'fit' and 'predict' of the estimator.
        is_has_fit_method = getattr(estimator, "fit")
        is_has_predict_method = getattr(estimator, "predict")

        # check if those class method are callable.
        is_has_fit_method = callable(is_has_fit_method)
        is_has_predict_method = callable(is_has_predict_method)

        # assumes it is an estimator if it has 'fit' and 'predict' methods.
        return is_has_fit_method and is_has_predict_method
    except AttributeError:
        return False


def _validate_single_datatype_features(x: pd.DataFrame) -> None:
    """
    Checks if all values in the feature belongs to the same datatype. If
    not, error 'MultipleDataTypesError will be raised.'

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
    Dataset (features only) that needed to be imputed.

    Raises
    ------
    MultipleDataTypesError
        If not all values in the feature belongs to the same datatype.
    """

    vectorized_type = np.vectorize(type)
    for c in x.columns:
        all_type = vectorized_type(x[c].dropna())
        if len(pd.unique(all_type)) > 1:
            raise MultipleDataTypesError(f"Feature {c} has more than one "
                                         f"datatype.")
