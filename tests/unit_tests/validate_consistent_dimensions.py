"""This module contains unit tests for `_validate_consistent_dimensions`."""


import unittest
import numpy as np
# noinspection PyProtectedMember
from src.missforest._validate import _validate_consistent_dimensions


class ValidateConsistentDimensions(unittest.TestCase):
    @staticmethod
    def test_validate_consistent_dimensions():
        """Tests if `_validate_consistent_dimensions` raises no error if two
        matrices that share the same shape are passed."""
        mat1 = np.zeros(shape=(2, 2))
        mat2 = np.ones(shape=(2, 2))
        _validate_consistent_dimensions(mat1, mat2)

    def test_validate_consistent_dimensions_inconsistent(self):
        """Tests if `_validate_consistent_dimensions` raises ValueError if
        two matrices that do not share the same shape are passed."""
        with self.assertRaises(ValueError):
            mat1 = np.zeros(shape=(2, 2))
            mat2 = np.ones(shape=(3, 3))
            _validate_consistent_dimensions(mat1, mat2)
