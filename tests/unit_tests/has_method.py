"""This model contains unit tests for `_has_method`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _has_method


class HasMethod(unittest.TestCase):
    def test_has_method_positive(self):
        """Tests if `_has_method` return True when an instance of an object
        that has class method `test` is passed."""
        class MockEstimator:
            def __init__(self):
                pass

            def test(self):
                pass

        mock_estimator = MockEstimator()
        self.assertTrue(_has_method(mock_estimator, "test"))

    def test_has_method_negative(self):
        """Tests if `_has_method` return False when an instance of an object
        that dose not has class method `test` is passed."""
        class MockEstimator:
            def __init__(self):
                pass

        mock_estimator = MockEstimator()
        self.assertFalse(_has_method(mock_estimator, "test"))


