"""This module contains label-encoding functions."""

from ._info import VERSION, AUTHOR

__all__ = ["_label_encoding", "_rev_label_encoding"]
__version__ = VERSION
__author__ = AUTHOR

import pandas as pd


def _label_encoding(x: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Performs label encoding on given features and the input mappings.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needs to be encoded.
    mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

    Returns
    -------
    x : pd.DataFrame of shape (n_samples, n_features)
        Label-encoded dataset (features only).
    """
    for c in mappings:
        x[c] = x[c].map(mappings[c]).astype(int)

    return x


def _rev_label_encoding(x: pd.DataFrame, rev_mappings: dict) -> pd.DataFrame:
    """Performs reverse label encoding on given features and the input
    reverse mappings.

    Parameters
    ----------
    x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needs to be imputed.
    rev_mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

    Returns
    -------
    x : pd.DataFrame of shape (n_samples, n_features)
        Reverse label-encoded dataset (features only).
    """
    for c in rev_mappings:
        x[c] = x[c].map(rev_mappings[c])

    return x
