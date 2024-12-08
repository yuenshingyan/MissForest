"""This module contains unit tests for `_validate_max_iter`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_max_iter


class ValidateMaxIter(unittest.TestCase):
    @staticmethod
    def test_max_iter_int():
        """Tests if `_validate_max_iter` raises no error when an int is passed.
        """
        _validate_max_iter(3)

    def test_max_iter_float(self):
        """Tests if `_validate_max_iter` raises ValueError when a float is
        passed."""
        with self.assertRaises(ValueError):
            _validate_max_iter(max_iter=3.5)

    def test_max_iter_str(self):
        """Tests if `_validate_max_iter` raises ValueError when a str is
        passed."""
        with self.assertRaises(ValueError):
            _validate_max_iter(max_iter="3.5")

    def test_max_iter_list(self):
        """Tests if `_validate_max_iter` raises ValueError when a list is
        passed."""
        with self.assertRaises(ValueError):
            _validate_max_iter(max_iter=[3.5])

    def test_max_iter_tuple(self):
        """Tests if `_validate_max_iter` raises ValueError when a tuple is
        passed."""
        with self.assertRaises(ValueError):
            # noinspection PyRedundantParentheses
            _validate_max_iter(max_iter=(3.5))

    def test_max_iter_set(self):
        """Tests if `_validate_max_iter` raises ValueError when a set is
        passed."""
        with self.assertRaises(ValueError):
            _validate_max_iter(max_iter={3.5})

    def test_max_iter_dict(self):
        """Tests if `_validate_max_iter` raises ValueError when a dict is
        passed."""
        with self.assertRaises(ValueError):
            _validate_max_iter(max_iter={3.5: 1})
