"""This module contains unit tests for `_validate_early_stopping`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_early_stopping


class ValidateEarlyStopping(unittest.TestCase):
    @staticmethod
    def test_validate_early_stopping_true():
        """Tests if `_validate_early_stopping` raises no ValueError when True
        is passed."""
        _validate_early_stopping(True)

    @staticmethod
    def test_validate_early_stopping_false():
        """Tests if `_validate_early_stopping` raises no ValueError when False
        is passed."""
        _validate_early_stopping(False)

    def test_validate_early_stopping_int(self):
        """Tests if `_validate_early_stopping` raises ValueError when an int
        is passed."""
        with self.assertRaises(ValueError):
            _validate_early_stopping(10)

    def test_validate_early_stopping_float(self):
        """Tests if `_validate_early_stopping` raises ValueError when a float
        is passed."""
        with self.assertRaises(ValueError):
            _validate_early_stopping(10.0)

    def test_validate_early_stopping_str(self):
        """Tests if `_validate_early_stopping` raises ValueError when a str
        is passed."""
        with self.assertRaises(ValueError):
            _validate_early_stopping("10")

    def test_validate_early_stopping_list(self):
        """Tests if `_validate_early_stopping` raises ValueError when a list
        is passed."""
        with self.assertRaises(ValueError):
            _validate_early_stopping([10])

    def test_validate_early_stopping_tuple(self):
        """Tests if `_validate_early_stopping` raises ValueError when tuple
        is passed."""
        with self.assertRaises(ValueError):
            # noinspection PyRedundantParentheses
            _validate_early_stopping((10))

    def test_validate_early_stopping_set(self):
        """Tests if `_validate_early_stopping` raises ValueError when a set
        is passed."""
        with self.assertRaises(ValueError):
            _validate_early_stopping({10})

    def test_validate_early_stopping_dict(self):
        """Tests if `_validate_early_stopping` raises ValueError when a dict
        is passed."""
        with self.assertRaises(ValueError):
            _validate_early_stopping({10: 1})
