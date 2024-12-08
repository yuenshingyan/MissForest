"""This module contains unit tests for `_validate_rgr`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_rgr


class ValidateRegressor(unittest.TestCase):
    @staticmethod
    def test_validate_rgr():
        """Tests if `_validate_rgr` raises no ValueError when an instance
        of `MockEstimator` that have class method `fit` and `predict` is
        passed."""
        class MockEstimator:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

        mock_estimator = MockEstimator()
        _validate_rgr(mock_estimator)

    def test_validate_rgr_no_fit(self):
        """Tests if `_validate_rgr` raises ValueError when an instance of
        `MockEstimator` that only have class method `predict` is passed."""
        class MockEstimator:
            def __init__(self):
                pass

            def predict(self):
                pass

        mock_estimator = MockEstimator()
        with self.assertRaises(ValueError):
            _validate_rgr(mock_estimator)

    def test_validate_rgr_no_predict(self):
        """Tests if `_validate_rgr` raises ValueError when an instance of
        `MockEstimator` that only have class method `fit` is passed."""
        class MockEstimator:
            def __init__(self):
                pass

            def fit(self):
                pass

        mock_estimator = MockEstimator()
        with self.assertRaises(ValueError):
            _validate_rgr(mock_estimator)

    def test_validate_rgr_is_none(self):
        """Tests if `_validate_rgr` raises ValueError when None is passed."""
        with self.assertRaises(ValueError):
            _validate_rgr(None)
