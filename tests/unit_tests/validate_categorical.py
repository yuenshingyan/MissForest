"""This module contains unit tests for `_validate_categorical`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_categorical


class ValidateCategorical(unittest.TestCase):
    @staticmethod
    def test_validate_categorical_list():
        """Tests if `_validate_categorical` raises no error when a list is
        passed."""
        _validate_categorical(["a"])

    @staticmethod
    def test_validate_categorical_none():
        """Tests if `_validate_categorical` raises ValueError if none is
        passed."""
        # noinspection PyTypeChecker
        _validate_categorical(None)

    def test_validate_categorical_tuple(self):
        """Tests if `_validate_categorical` raises no error when a tuple is
        passed."""
        with self.assertRaises(ValueError):
            # noinspection PyRedundantParentheses
            _validate_categorical(("a", "b"))

    def test_validate_categorical_set(self):
        """Tests if `_validate_categorical` raises no error when a set is
        passed."""
        with self.assertRaises(ValueError):
            _validate_categorical({"a"})

    def test_validate_categorical_dict(self):
        """Tests if `_validate_categorical` raises no error when a dict is
        passed."""
        with self.assertRaises(ValueError):
            _validate_categorical({"a": 1})

    def test_validate_categorical_int(self):
        """Tests if `_validate_categorical` raises ValueError if int is
        passed."""
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            _validate_categorical(1)

    def test_validate_categorical_float(self):
        """Tests if `_validate_categorical` raises ValueError if float is
        passed."""
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            _validate_categorical(1.0)

    def test_validate_categorical_str(self):
        """Tests if `_validate_categorical` raises ValueError if str is
        passed."""
        with self.assertRaises(ValueError):
            _validate_categorical("a")
