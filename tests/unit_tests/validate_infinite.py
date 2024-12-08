"""This module contains unit tests for `_validate_infinite`."""


import unittest
import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from src.missforest._validate import _validate_infinite


class ValidateInfinite(unittest.TestCase):
    @staticmethod
    def test_validate_infinite_non_inf():
        """Tests if `_validate_infinite` raises no error when a pandas
        dataframe that contains no infinite value in it."""
        df = pd.DataFrame(data=[
            [1, 2],
            [3, 4]
        ])
        _validate_infinite(df)

    def test_validate_infinite_inf(self):
        """Tests if `_validate_infinite` raises ValueError if when a pandas
        dataframe contains infinite value in it."""
        with self.assertRaises(ValueError):
            df = pd.DataFrame(data=[
                [1, 2],
                [np.inf, 4]
            ])
            _validate_infinite(df)
