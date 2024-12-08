"""This module contains unit tests for `_validate_empty_feature`."""


import unittest
import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from src.missforest._validate import _validate_empty_feature


class ValidateEmptyFeature(unittest.TestCase):
    @staticmethod
    def test_validate_empty_feature():
        """Tests if `_validate_empty_feature` raises no error when a pandas
        dataframe that contains no empty feature is passed."""
        df = pd.DataFrame(data=[
            [1, 2],
            [3, 4]
        ])
        _validate_empty_feature(df)

    def test_validate_empty_feature_positive(self):
        """Tests if `_validate_empty_feature` raises ValueError when a pandas
        dataframe that contains empty feature is passed."""
        with self.assertRaises(ValueError):
            df = pd.DataFrame(data=[
                [np.nan, 2],
                [np.nan, 4]
            ])
            _validate_empty_feature(df)
