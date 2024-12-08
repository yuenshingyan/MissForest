"""This module contains unit tests for `_validate_cat_var_consistency`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_cat_var_consistency


class ValidateCategoricalVariableConsistency(unittest.TestCase):
    def test_validate_cat_var_consistency(self):
        """Tests if `_validate_cat_var_consistency` raises no error when
        passed `features` is subset of `categorical`."""
        with self.assertRaises(ValueError):
            features = ["a", "b", "c"]
            categorical = ["a", "b", "c", "d"]
            _validate_cat_var_consistency(features, categorical)

    @staticmethod
    def test_validate_cat_var_consistency_identical():
        """Tests if `_validate_cat_var_consistency` raises no error when
        passed `features` is subset of `categorical`."""
        features = ["a", "b", "c", "d"]
        categorical = ["a", "b", "c", "d"]
        _validate_cat_var_consistency(features, categorical)

    @staticmethod
    def test_validate_cat_var_consistency_superset():
        """Tests if `_validate_cat_var_consistency` raises ValueError when
        passed `features` is superset of `categorical`."""
        features = ["a", "b", "c", "d"]
        categorical = ["a", "b", "c"]
        _validate_cat_var_consistency(features, categorical)
