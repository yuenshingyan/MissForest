"""This module contains unit tests for `_is_numerical_matrix`."""


import unittest
import numpy as np
# noinspection PyProtectedMember
from src.missforest._validate import _is_numerical_matrix


class IsNumericalMatrix(unittest.TestCase):
    def test_is_numerical_matrix_positive(self):
        """Tests if `_is_numerical_matrix` returns True when a numerical
        matrix is passed."""
        rand_size = np.random.randint(low=1, high=10, size=2)
        rand_mat = np.random.random(size=rand_size)
        self.assertTrue(_is_numerical_matrix(rand_mat))

    def test_is_numerical_matrix_negative(self):
        """Tests if `_is_numerical_matrix` raises False when a non-fully
        numerical matrix is passed."""
        rand_mat = np.array([["a", 2], [3, 4]])
        self.assertFalse(_is_numerical_matrix(rand_mat))
