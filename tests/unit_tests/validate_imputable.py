"""This module contains unit tests for `_validate_imputable`."""


import unittest
import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from src.missforest._validate import _validate_imputable


class ValidateImputable(unittest.TestCase):
    @staticmethod
    def test_validate_imputable():
        """Tests if `_validate_imputable` raises no error if passed pandas
        dataframe has at least one missing value."""
        df = pd.DataFrame(data=[
            ["a", "b"],
            [np.nan, "d"]
        ])
        _validate_imputable(df)

    def test_validate_imputable_no_missing_value(self):
        """Tests if `_validate_imputable` raises ValueError if passed pandas
        dataframe has no missing value."""
        with self.assertRaises(ValueError):
            df = pd.DataFrame(data=[
                ["a", "b"],
                ["c", "d"]
            ])
            _validate_imputable(df)
