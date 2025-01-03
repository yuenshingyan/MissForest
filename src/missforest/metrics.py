"""This module contains all functions for computing metrics."""

from ._info import VERSION, AUTHOR

__all__ = ["pfc", "nrmse"]
__version__ = VERSION
__author__ = AUTHOR

import numpy as np
import pandas as pd
from ._validate import _is_numerical_matrix


def pfc(x_true: pd.DataFrame, x_imp: pd.DataFrame, n_missing: int) -> float:
    """Compute and return the proportion of falsely classified (PFC) of two
    categorical matrices `x_true` and `x_imp`.

    Parameters
    ----------
    x_true : pd.DataFrame
        Complete data matrix.
    x_imp : pd.DataFrame
        Imputed data matrix.
    n_missing : int
        Total number of missing values in the categorical variables.

    Returns
    -------
    float
        Proportion of falsely classified (PFC) of two different categorical
        imputation matrices.
    """
    if x_true.shape != x_imp.shape:
        raise ValueError(
            f"Argument `x_true` has a shape of {x_true.shape}, "
            f"which is not identical to shape of `x_imp` {x_imp.shape}."
        )

    if not isinstance(n_missing, int):
        raise ValueError("Argument `n_missing` must be int.")

    if n_missing < 0:
        raise ValueError("Argument `n_missing` must be positive.")

    if n_missing == 0:
        return 0.0

    return np.sum(np.sum(x_true != x_imp, axis=1)) / n_missing


def nrmse(x_true: pd.DataFrame, x_imp: pd.DataFrame) -> float:
    """Compute and return the normalized root mean squared error (NRMSE) two
    different numerical matrices `x_true` and `x_imp`.

    Parameters
    ----------
    x_true : pd.DataFrame of shape (n_samples, n_features)
        Complete data matrix.
    x_imp : pd.DataFrame of shape (n_samples, n_features)
        Imputed data matrix.

    Returns
    -------
    float
        Normalized root mean squared error (NRMSE) two different numerical
        imputation matrices.
    """
    if x_true.shape != x_imp.shape:
        raise ValueError(
            f"Argument `x_true` has a shape of {x_true.shape}, "
            f"which is not identical to the shape of `x_imp` {x_imp.shape}.")

    if not _is_numerical_matrix(x_true):
        raise ValueError("Argument `x_true` must be fully numerical.")

    if not _is_numerical_matrix(x_imp):
        raise ValueError("Argument `x_imp` must be fully numerical.")

    return np.sum(np.sum(
        (x_true - x_imp) ** 2, axis=0
    ), axis=0) / np.sum(np.sum(x_true ** 2, axis=0), axis=0)
