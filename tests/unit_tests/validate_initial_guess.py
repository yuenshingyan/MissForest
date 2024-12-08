"""This module contains unit tests for `_validate_initial_guess`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_initial_guess


class ValidateInitialGuess(unittest.TestCase):
    @staticmethod
    def test_initial_guess_mean():
        """Tests if `_validate_initial_guess` raises no ValueError when `mean`
        is passed."""
        _validate_initial_guess("mean")

    @staticmethod
    def test_initial_guess_median():
        """Tests if `_validate_initial_guess` raises no ValueError when `median`
        is passed."""
        _validate_initial_guess("median")

    def test_initial_guess_other_str(self):
        """Tests if `_validate_initial_guess` raises ValueError when other str
        (rather than `mean` or `median`) is passed."""
        with self.assertRaises(ValueError):
            _validate_initial_guess("mode")

    def test_initial_guess_int(self):
        """Tests if `_validate_initial_guess` raises ValueError when an integer
        is passed."""
        with self.assertRaises(ValueError):
            _validate_initial_guess(10)

    def test_initial_guess_float(self):
        """Tests if `_validate_initial_guess` raises ValueError when a float
        is passed."""
        with self.assertRaises(ValueError):
            _validate_initial_guess(10.0)

    def test_initial_guess_list(self):
        """Tests if `_validate_initial_guess` raises ValueError when a list
        is passed."""
        with self.assertRaises(ValueError):
            _validate_initial_guess([10])

    def test_initial_guess_tuple(self):
        """Tests if `_validate_initial_guess` raises ValueError when a tuple
        is passed."""
        with self.assertRaises(ValueError):
            # noinspection PyRedundantParentheses
            _validate_initial_guess((10))

    def test_initial_guess_set(self):
        """Tests if `_validate_initial_guess` raises ValueError when a set
        is passed."""
        with self.assertRaises(ValueError):
            _validate_initial_guess({10})

    def test_initial_guess_dict(self):
        """Tests if `_validate_initial_guess` raises ValueError when a dict
        is passed."""
        with self.assertRaises(ValueError):
            _validate_initial_guess({10: 1})
