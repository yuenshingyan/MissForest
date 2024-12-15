"""This module contains all validation functions."""

from ._info import VERSION, AUTHOR

__all__ = [
    "_validate_verbose",
    "_validate_clf",
    "_validate_rgr",
    "_validate_initial_guess",
    "_validate_max_iter",
    "_validate_early_stopping",
    "_validate_feature_dtype_consistency",
    "_is_numerical_matrix",
    "_validate_2d",
    "_validate_consistent_dimensions",
    "_validate_cat_var_consistency",
    "_validate_categorical",
    "_validate_infinite",
    "_validate_empty_feature",
    "_validate_imputable",
]
__version__ = VERSION
__author__ = AUTHOR


from typing import Any, Union, Iterable
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from .errors import FeaturesDataTypeInconsistentError


def _validate_verbose(verbose: int):
    """Checks if argument `verbose` is either 0, 1 or 2.

    Parameters
    ----------
    verbose : int
        Verbosity to be checked.
    """

    if verbose not in (0, 1, 2):
        raise ValueError("Argument `verbose` must be either 0, 1, or 2.")


def _validate_imputable(x: pd.DataFrame):
    """Checks if argument `x` contains at least one missing value.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataframe to be checked.

    Raises
    ------
    ValueError
        - If argument `x` does not contain any missing value.
    """
    if not x.isnull().any().any():
        raise ValueError(
            "Argument `x` must contains at least one missing value.")


def _validate_empty_feature(x: pd.DataFrame):
    """Checks if argument `x` has any features that contains no values.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataframe to be checked.

    Raises
    ------
    ValueError
        - If argument `x` has any features that contains no values.
    """
    if x.isnull().all().any():
        raise ValueError(
            "One or more columns have all missing values in argument `x`.")


def _validate_infinite(x: pd.DataFrame):
    """Checks if argument `x` contains infinite values.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataframe to be checked

    Raises
    ------
    ValueError
        - If argument `x` contains infinite values.
    """
    if np.any(np.isinf(x)):
        raise ValueError("Argument `x` must not contains infinite values.")


def _validate_categorical(categorical: Union[None, list[Any]]):
    """Checks if argument `categorical` iterable.

    Parameters
    ----------
    categorical : Union[None, list[Any]]
        Categorical variables defined by users.

    Raises
    ------
    ValueError
        - If argument `categorical` is iterable.
    """
    if not isinstance(categorical, list) and categorical is not None:
        raise ValueError("Argument `categorical` must be None or list.")


def _validate_cat_var_consistency(features: Iterable, categorical: Iterable):
    """Checks if `features` is a subset of `categorical`.

    Parameters
    ----------
    features : Iterable
        All features in `x`.
    categorical : Iterable
        Categorical variables defined by users.

    Raises
    ------
    ValueError
        - If `features` is not a subset of `categorical`.
    """
    if not set(categorical).issubset(features):
        raise ValueError("Not all features in argument `categorical` "
                         "existed in `x` columns.")


def _validate_consistent_dimensions(mat1: pd.DataFrame, mat2: pd.DataFrame):
    """Checks if arguments `mat1` and `mat2` share the same shape.

    Parameters
    ----------
    mat1: pd.DataFrame
        First matrix.
    mat2: pd.DataFrame
        Second matrix.

    Raises
    ------
    ValueError
        - If argument `mat1` and `mat2` does not share the same shape.
    """
    if mat1.shape != mat2.shape:
        raise ValueError(
            "Argument `mat1` and `mat2` does not share the same shape.")


def _validate_2d(x: Any):
    """Checks if argument `array` is 2D.

    Parameters
    ----------
    x : Any
        Array to be checked.

    Raises
    ------
    ValueError
        - If argument `x` is not 2D.
    """
    if len(x.shape) != 2:
        raise ValueError("Argument `x` must be 2D array.")


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


def _validate_max_iter(max_iter: int):
    """Checks if argument `max_iter` is int, and if `max_iter` is less than
    or equals to 1.

    Parameters
    ----------
    max_iter : Any
        Maximum number of iterations in MissForest.

    Raises
    ------
    ValueError
        - If argument `max_iter` is not int.
        - If argument `max_iter` is less than or equals to 1.
    """
    if not isinstance(max_iter, int):
        raise ValueError("Argument `max_iter` must be int.")

    if max_iter <= 1:
        raise ValueError("Argument `max_iter` must be greater than 1.")


def _validate_early_stopping(early_stopping: bool):
    """Checks if argument `early_stopping` is bool.

    Parameters
    ----------
    early_stopping : Any
        Boolean that determines if early stopping mechanism is enabled.

    Raises
    ------
    ValueError
        - If argument `early_stopping` is not bool.
    """
    if not isinstance(early_stopping, bool):
        raise ValueError("Argument `early_stopping` must be bool.")


def _validate_initial_guess(initial_guess: str):
    """Checks if `initial_guess` is str, and it `initial_guess` are either
    `median` or `mean`.

    Parameters
    ----------
    initial_guess : str
        Initial guess.

    Raises
    ------
    ValueError
        - If argument `initial_guess` is not str.
        - If argument `initial_guess` is neither `median` nor `mean`.
    """
    if not isinstance(initial_guess, str):
        raise ValueError("Argument `initial_guess` must be str.")

    if initial_guess not in ("median", "mean"):
        raise ValueError(
            "Argument `initial_guess` can only be `median` or `mean`.")


def _validate_clf(clf: Any):
    """Checks if argument `clf` has methods `fit` and `predict`.

    Parameters
    ----------
    clf : Any
        Regressor or estimator to be checked.

    Raises
    ------
    ValueError
        - If argument `clf` is None.
        - If argument `clf` has no method `fit`.
        - If argument `clf` has no method `predict`.
    """
    if clf is None:
        raise ValueError("Argument `clf` is None.")

    if (
            not _has_method(clf, "fit") or
            not _has_method(clf, "predict")
    ):
        raise ValueError("Argument `clf` only accept classifier that has "
                         "class methods `fit` and `predict`.")


def _validate_rgr(rgr: Any):
    """Checks if argument `rgr` has methods `fit` and `predict`.

    Parameters
    ----------
    rgr : Any
        Regressor or estimator to be checked.

    Raises
    ------
    ValueError
        - If argument `rgr` is None.
        - If argument `rgr` has no method `fit`.
        - If argument `rgr` has no method `predict`.
    """
    if rgr is None:
        raise ValueError("Argument `rgr` is None.")

    if (
            not _has_method(rgr, "fit") or
            not _has_method(rgr, "predict")
    ):
        raise ValueError("Argument `rgr` only accept regressor that has "
                         "class methods `fit` and `predict`.")


def _has_method(estimator: Union[Any, BaseEstimator], method: str) -> bool:
    """Helper function that checks if argument `estimator` has method
    `method`.

    Parameters
    ----------
    estimator : Union[Any, BaseEstimator]
        Estimator being validated.
    method : str
        Method being validated.

    Returns
    -------
    bool
        - True, if argument `estimator` has method  `method`.
        - False, if argument `estimator` has no method  `method`.
    """
    try:
        return callable(getattr(estimator, method))
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
        - If not all datatypes are homogeneous, for each feature.
    """
    vectorized_type = np.vectorize(type)
    for c in x.columns:
        all_type = vectorized_type(x[c].dropna())
        if len(pd.unique(all_type)) > 1:
            raise FeaturesDataTypeInconsistentError(
                f"Multiple data types found in feature `{c}`.")
