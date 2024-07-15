"""This module contains all functions for computing metrics."""

from ._info import VERSION, AUTHOR

__all__ = ["pfc", "nrmse"]
__version__ = VERSION
__author__ = AUTHOR

import numpy as np
import pandas as pd


def pfc(x_imp_cat_curr: pd.DataFrame, x_imp_cat_prev: pd.DataFrame) -> float:
    """Compute and return the proportion of falsely classified (PFC) of two
    different categorical imputation matrices.

    Parameters
    ----------
    x_imp_cat_curr : pd.DataFrame
        Latest categorical imputation matrix.
    x_imp_cat_prev : pd.DataFrame
        Before last categorical imputation matrix.

    Returns
    -------
    float
        Proportion of falsely classified (PFC) of two different categorical
        imputation matrices.
    """
    if len(x_imp_cat_curr) != len(x_imp_cat_prev):
        raise ValueError(f"Argument `x_imp_cat_curr` has a length of "
                         f"{len(x_imp_cat_curr)}, which is not identical to "
                         f"length of {len(x_imp_cat_prev)}.")
    return (
            np.sum(np.sum(x_imp_cat_curr != x_imp_cat_prev, axis=1)) /
            len(x_imp_cat_curr)
    )


def nrmse(x_imp_num_curr: pd.DataFrame, x_imp_num_prev: pd.DataFrame) -> float:
    """Compute and return the normalized root mean squared error (NRMSE) two
    different numerical imputation matrices.

    Parameters
    ----------
    x_imp_num_curr : pd.DataFrame
        Latest numerical imputation matrix.
    x_imp_num_prev : pd.DataFrame
        Before last numerical imputation matrix.

    Returns
    -------
    float
        Normalized root mean squared error (NRMSE) two different numerical
        imputation matrices.
    """
    if len(x_imp_num_curr) != len(x_imp_num_prev):
        raise ValueError(f"Argument `x_imp_cat_curr` has a length of "
                         f"{len(x_imp_num_curr)}, which is not identical to "
                         f"length of {len(x_imp_num_prev)}.")
    return np.sum(np.sum(
        (x_imp_num_curr - x_imp_num_prev) ** 2, axis=0
    ), axis=0) / np.sum(np.sum(x_imp_num_curr ** 2, axis=0), axis=0)
