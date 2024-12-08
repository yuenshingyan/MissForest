"""This module contains unit tests for `_validate_clf`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_clf


class ValidateClassifier(unittest.TestCase):
    @staticmethod
    def test_validate_clf():
        """Tests if `_validate_clf` raises no ValueError when an instance of
        `MockEstimator` that have class method `fit` and `predict` is passed.
        """
        class MockEstimator:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

        mock_estimator = MockEstimator()
        _validate_clf(mock_estimator)

    def test_validate_clf_no_fit(self):
        """Tests if `_validate_clf` raises ValueError when an instance of
        `MockEstimator` that have class method `predict` is passed."""
        class MockEstimator:
            def __init__(self):
                pass

            def predict(self):
                pass

        mock_estimator = MockEstimator()
        with self.assertRaises(ValueError):
            _validate_clf(mock_estimator)

    def test_validate_clf_no_predict(self):
        """Tests if `_validate_clf` raises ValueError when an instance of
        `MockEstimator` that have class method `fit` is passed."""
        class MockEstimator:
            def __init__(self):
                pass

            def fit(self):
                pass

        mock_estimator = MockEstimator()
        with self.assertRaises(ValueError):
            _validate_clf(mock_estimator)

    def test_validate_clf_is_none(self):
        """Tests if `_validate_clf` raises ValueError when None is passed."""
        with self.assertRaises(ValueError):
            _validate_clf(None)
