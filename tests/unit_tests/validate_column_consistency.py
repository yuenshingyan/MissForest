"""This module contains unit tests for `_validate_column_consistency`."""


import unittest
# noinspection PyProtectedMember
from src.missforest._validate import _validate_column_consistency


class ValidateColumnConsistency(unittest.TestCase):
    @staticmethod
    def test_validate_column_consistency():
        """Tests if no ValueError is raised if `existing` and `new` are
        consistent."""
        existing = {1, 2, 3}
        new = {1, 2, 3}
        _validate_column_consistency(existing, new)

    def test_validate_column_consistency_negative(self):
        """Tests if ValueError is raised if `existing` and `new` are not
        consistent."""
        with self.assertRaises(ValueError):
            existing = {1, 2, 3}
            new = {1, 2}
            _validate_column_consistency(existing, new)
