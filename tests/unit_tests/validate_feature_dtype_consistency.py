"""This module contains unit tests for `_validate_feature_dtype_consistency`.
"""


import unittest
import pandas as pd
# noinspection PyProtectedMember
from src.missforest._validate import _validate_feature_dtype_consistency
from src.missforest.errors import FeaturesDataTypeInconsistentError


class FeatureDataType(unittest.TestCase):
    def test_validate_feature_dtype_consistency_negative(self):
        """Tests if `_validate_single_datatype_features` raises
        `FeaturesDataTypeInconsistentError` when a pandas dataframe that
        contains inconsistent dtype feature."""
        df = pd.DataFrame(
            data={'mixed_column': ['123', 456, True, 'hello', None]})
        with self.assertRaises(FeaturesDataTypeInconsistentError):
            _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_int():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all int."""
        df = pd.DataFrame(
            data={'column': [1, 2, 3]})
        _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_float():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all float."""
        df = pd.DataFrame(
            data={'column': [1.0, 2.0, 3.0]})
        _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_str():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all str."""
        df = pd.DataFrame(
            data={'column': ["1", "2", "3"]})
        _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_list():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all list."""
        df = pd.DataFrame(
            data={'column': [[1], [2], [3]]})
        _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_tuple():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all tuple."""
        df = pd.DataFrame(
            data={'column': [(1), (2), (3)]})
        _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_set():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all set."""
        df = pd.DataFrame(
            data={'column': [{1}, {2}, {3}]})
        _validate_feature_dtype_consistency(df)

    @staticmethod
    def test_validate_feature_dtype_consistency_dict():
        """Tests if `_validate_single_datatype_features` raises no
        `FeaturesDataTypeInconsistentError` if all dtypes of the same column
        in a pandas dataframe are all dict."""
        df = pd.DataFrame(
            data={'column': [{1: 1}, {2: 1}, {3: 1}]})
        _validate_feature_dtype_consistency(df)
