"""This module contains unit tests for `_validate_verbose`."""


import unittest

# noinspection PyProtectedMember
from src.missforest._validate import _validate_verbose


class ValidateVerbose(unittest.TestCase):
    @staticmethod
    def test_validate_verbose_0():
        """Tests if `_validate_verbose` raises no error if 0 were passed."""
        _validate_verbose(0)

    @staticmethod
    def test_validate_verbose_1():
        """Tests if `_validate_verbose` raises no error if 1 were passed."""
        _validate_verbose(1)

    @staticmethod
    def test_validate_verbose_2():
        """Tests if `_validate_verbose` raises no error if 2 were passed."""
        _validate_verbose(2)

    def test_validate_verbose_3(self):
        """Tests if `_validate_verbose` raises no error if 3 were passed."""
        with self.assertRaises(ValueError):
            _validate_verbose(3)
