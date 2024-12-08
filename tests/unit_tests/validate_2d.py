"""This module contains unit tests for `_validate_2d`."""


import unittest
import numpy as np
# noinspection PyProtectedMember
from src.missforest._validate import _validate_2d


class Validate2D(unittest.TestCase):
    def test_validate_2d_array_1d(self):
        """Tests if `_validate_2d` raises ValueError when an 1D array is
        passed."""
        with self.assertRaises(ValueError):
            array_1d = np.zeros(shape=(1, ))
            _validate_2d(array_1d)

    @staticmethod
    def test_validate_2d_array_2d():
        """Tests if `_validate_2d` raises no error when a 2D array is passed.
        """
        array_2d = np.zeros(shape=(1, 2))
        _validate_2d(array_2d)

    def test_validate_2d_array_3d(self):
        """Tests if `_validate_2d` raises ValueError when a 3D array is
        passed."""
        with self.assertRaises(ValueError):
            array_3d = np.zeros(shape=(1, 2, 3))
            _validate_2d(array_3d)
