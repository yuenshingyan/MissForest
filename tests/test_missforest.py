"""This module contains class 'TestMissForest'."""

__author__ = "Yuen Shing Yan Hindy"

import unittest
from src.missforest.missforest import MissForest
import pandas as pd
import numpy as np
from scipy.stats import norm, binom
from sklearn.model_selection import train_test_split


class TestMissForest(unittest.TestCase):
    """Class 'TestMissForest' is used for to unit-tests all class methods in
    'TestMissForest'."""

    def setUp(self):
        """Class method 'setUp' is a special method that is automatically
        called before each test method is executed. It is used to set up
        'MissForest' instance that is shared across multiple tests."""

        self.missforest = MissForest()

        while True:
            # make synthetic datasets
            # seed to follow along
            np.random.seed(1234)

            # generate 1000 data points
            n = np.arange(1000)

            # helper function for this data
            vary = lambda v: np.random.choice(np.arange(v))

            # create correlated, random variables
            a = 2
            b = 1 / 2
            eps = np.array([norm(0, vary(50)).rvs() for _ in n])
            y = (a + b * n + eps) / 100
            x = (n + norm(10, vary(250)).rvs(len(n))) / 100

            # add missing values
            y[binom(1, 0.4).rvs(len(n)) == 1] = np.nan

            # convert to dataframe
            df = pd.DataFrame({"y": y, "x": x})

            self.train, self.test = train_test_split(df, test_size=.3)

            if (
                    self.train.isnull().sum().sum() > 0 and
                    self.test.isnull().sum().sum() > 0
            ):
                break

    def test_guarding_logic_initial_guess_str(self):
        """Class method 'test_guarding_logic_initial_guess_str' tests if
        'MissForest' can be instantiated properly with argument 'initial_guess'
        being 'mean' and 'median'."""

        MissForest(initial_guess="mean")
        MissForest(initial_guess="median")

    def test_guarding_logic_initial_guess_non_str(self):
        """Class method 'test_guarding_logic_initial_guess_non_str' test if
        'ValueError' will be raised if 'MissForest' is improperly instantiated
        with argument 'initial_guess' being integer."""

        with self.assertRaises(ValueError):
            MissForest(initial_guess=10)

    def test_guarding_logic_initial_guess_not_mean_or_median(self):
        """Class method 'test_guarding_logic_initial_guess_not_mean_or_median'
         test if  'ValueError' will be raised if 'MissForest' is improperly
         instantiated  with argument 'initial_guess' being string but not
         'mean' or 'median'."""

        with self.assertRaises(ValueError):
            MissForest(initial_guess="mode")

    def test_guarding_logic_max_iter_int(self):
        """Class method 'test_guarding_logic_max_iter_int' tests if
        'MissForest' can be instantiated properly with argument 'max_iter'
        being datatype 'int'."""

        MissForest(max_iter=3)

    def test_guarding_logic_max_iter_non_int(self):
        """Class method 'test_guarding_logic_max_iter_non_int' test if
        'ValueError' will be raised if 'MissForest' is improperly instantiated
        with argument 'max_iter' being datatype 'float'."""

        with self.assertRaises(ValueError):
            MissForest(max_iter=3.5)

    def test_is_estimator(self):
        """Clas method 'test_is_estimator' test if class method '_is_estimator'
         of 'MissForest' returns 'True' if argument is a class object that has
          'fit' and 'predict' methods."""

        class TestEstimator:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

        test_estimator = TestEstimator()
        self.assertTrue(self.missforest._is_estimator(test_estimator))

    def test_is_estimator_no_fit(self):
        """Clas method 'test_is_estimator_no_fit' test if class method
        '_is_estimator' of 'MissForest' returns 'False' if argument is a class
         object that only has 'predict' method."""

        class TestEstimator:
            def __init__(self):
                pass

            def predict(self):
                pass

        test_estimator = TestEstimator()
        self.assertFalse(self.missforest._is_estimator(test_estimator))

    def test_is_estimator_no_predict(self):
        """Clas method 'test_is_estimator_no_predict' test if class method
        '_is_estimator' of 'MissForest' returns 'False' if argument is a class
         object that only has 'fit' method."""

        class TestEstimator:
            def __init__(self):
                pass

            def fit(self):
                pass

        test_estimator = TestEstimator()
        self.assertFalse(self.missforest._is_estimator(test_estimator))

    def test_is_estimator_no_fit_or_predict(self):
        """Clas method 'test_is_estimator_no_predict' test if class method
        '_is_estimator' of 'MissForest' returns 'False' if argument is a class
         object that has no 'fit' or 'predict' methods."""

        class TestEstimator:
            def __init__(self):
                pass

        test_estimator = TestEstimator()

        self.assertFalse(self.missforest._is_estimator(test_estimator))

    def test_is_estimator_none(self):
        """Clas method 'test_is_estimator_no_predict' test if class method
        '_is_estimator' of 'MissForest' returns 'False' if argument is None."""

        test_estimator = None
        self.assertFalse(self.missforest._is_estimator(test_estimator))

    def test_get_missing_rows(self):
        """Class method 'test_get_missing_rows' test if '_get_missing_rows' of
        'MissForest' can correctly gather the index of any rows that has
        missing values."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, 2, 3, None]
        })

        mf = MissForest()
        mf._get_missing_rows(df)
        expected_result = {'A': [2], 'B': [0], 'C': [3]}
        self.assertEqual(mf._missing_row, expected_result)

    def test_get_obs_row(self):
        """Class method 'test_get_obs_row' test if '_get_obs_row' of
        'MissForest' correctly gather the rows of any rows that do not have any
         missing values."""

        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, None, 9, 10],
            'C': [11, 12, 13, None, 15]
        })

        # Call the method and get the result
        self.missforest._get_obs_row(df)

        # Check that the result is a pd.Index object
        self.assertIsInstance(self.missforest._obs_row, pd.Index)

        # Check that the result contains the indexes of the rows with no
        # missing values
        self.assertEqual(self.missforest._obs_row.tolist(), [0, 1, 4])

    def test_get_map_and_rev_map(self):
        """Class method 'test_get_map_and_rev_map' test if
        'test_get_map_and_rev_map' correctly construct dictionaries for label
         encoding and reverse-label encoding."""

        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        self.missforest._get_map_and_rev_map(df, ("A", "B", "C"))

        expected_mappings = {
            'A': {'a': 0, 'b': 1},
            'B': {'b': 0, 'c': 1},
            'C': {'a': 0, 'b': 1, 'c': 2}
        }
        expected_rev_mappings = {
            'A': {0: 'a', 1: 'b'},
            'B': {0: 'b', 1: 'c'},
            'C': {0: 'a', 1: 'b', 2: 'c'}
        }

        self.assertEqual(self.missforest._mappings, expected_mappings)
        self.assertEqual(self.missforest._rev_mappings, expected_rev_mappings)

    def test_get_initials_mean(self):
        """Class method '_get_initials' test if the initial imputations values
        are calculated and stored, under the circumstance that argument
        'initial_guess' of MissForest is set to 'mean'."""

        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['a', 'a', 'b', 'c']
        })

        self.missforest.initial_guess = 'mean'
        self.missforest._get_initials(df, ("B"))
        self.assertEqual(self.missforest._initials["A"], 2.5)
        self.assertEqual(self.missforest._initials["B"], "a")

    def test_get_initials_median(self):
        """Class method '_get_initials' test if the initial imputations values
        are calculated and stored, under the circumstance that argument
        'initial_guess' of MissForest is set to 'median'."""

        df = pd.DataFrame({
            'A': [1, 2, 2, 4],
            'B': ['a', 'a', 'b', 'c']
        })

        self.missforest.initial_guess = 'median'
        self.missforest._get_initials(df, ("B"))
        self.assertEqual(self.missforest._initials["A"], 2)
        self.assertEqual(self.missforest._initials["B"], "a")

    def test_initial_imputation_mean(self):
        """Class method 'test_initial_imputation_mean' test if the missing
        values are imputed correctly under the circumstance that argument
        'initial_guess' of MissForest is set to 'mean'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ["a", "a", "b", "c"]
        })

        self.missforest.initial_guess = 'mean'
        self.missforest._get_initials(df, ("C"))
        X_imp = self.missforest._initial_imputation(df)
        df["A"].fillna(df["A"].mean(), inplace=True)
        df["B"].fillna(df["B"].mean(), inplace=True)
        df["C"].fillna(df["C"].mode(), inplace=True)
        pd.testing.assert_frame_equal(X_imp, df, check_dtype=False)

    def test_initial_imputation_median(self):
        """Class method 'test_initial_imputation_median' test if the missing
        values are imputed correctly under the circumstance that argument
        'initial_guess' of MissForest is set to 'median'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ["a", "a", "b", "c"]
        })

        self.missforest.initial_guess = 'median'
        self.missforest._get_initials(df, ("C"))
        X_imp = self.missforest._initial_imputation(df)
        df["A"].fillna(df["A"].median(), inplace=True)
        df["B"].fillna(df["B"].median(), inplace=True)
        df["C"].fillna(df["C"].mode(), inplace=True)
        pd.testing.assert_frame_equal(X_imp, df, check_dtype=False)

    def test_get_initials_non_existing_feature(self):
        """Class method 'test_initial_imputation_mean' test if the missing
        values are imputed correctly with the mean values if argument
        'initial_guess' is set to 'mean'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ["a", "a", "b", "c"]
        })

        self.missforest.initial_guess = 'mean'
        with self.assertRaises(ValueError):
            self.missforest._get_initials(df, ("D"))

    def test_initial_imputation_mode(self):
        """Class method 'test_initial_imputation_mode' test if 'ValueError' is
         raised if values other than 'mean' or 'median' is passed to class
         method '_initial_imputation_mode' of 'MissForest'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ["a", "a", "b", "c"]
        })

        self.missforest.initial_guess = 'mode'
        with self.assertRaises(ValueError):
            self.missforest._get_initials(df, ("C"))

    def test_label_encoding(self):
        """Class method 'test_label_encoding' test if '_label_encoding' of
         'MissForest' correctly label encode the values in the pandas
         dataframe."""

        df = pd.DataFrame({
            'col1': ['A', 'B', 'A', 'A', 'B'],
            'col2': [1, 2, 3, 1, 2],
            'col3': ['X', 'Y', 'X', 'X', 'Y']
        })

        # Create a dictionary with mappings for label encoding
        mappings = {
            'col1': {'A': 0, 'B': 1},
            'col3': {'X': 0, 'Y': 1}
        }

        # Call the _label_encoding method
        result = self.missforest._label_encoding(df, mappings)

        # Create the expected output
        expected = pd.DataFrame({
            'col1': [0, 1, 0, 0, 1],
            'col2': [1, 2, 3, 1, 2],
            'col3': [0, 1, 0, 0, 1]
        })

        # Assert that the result is equal to the expected output
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_rev_label_encoding(self):
        """Class method 'test_rev_label_encoding' test if '_rev_label_encoding'
         of 'MissForest' correctly reverse label encode the values in the
          pandas dataframe."""

        # Create a DataFrame with missing values
        df = pd.DataFrame({
            'col1': [0, 1, 0, 0, 1],
            'col2': [1, 2, 3, 1, 2],
            'col3': [0, 1, 0, 0, 1]
        })

        # Create a dictionary with mappings for label encoding
        mappings = {
            'col1': {0: "A", 1: "B"},
            'col3': {0: "X", 1: "Y"}
        }

        # Call the _label_encoding method
        result = self.missforest._rev_label_encoding(df, mappings)

        # Create the expected output
        expected = pd.DataFrame({
            'col1': ['A', 'B', 'A', 'A', 'B'],
            'col2': [1, 2, 3, 1, 2],
            'col3': ['X', 'Y', 'X', 'X', 'Y']
        })

        # Assert that the result is equal to the expected output
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_integration1(self):
        """Class method test_integration1 test if MissForest can run properly
        when its 'fit' and 'transform' methods are called."""

        mf = MissForest()
        mf.fit(self.train)
        train_imputed = mf.transform(self.test)
        test_imputed = mf.transform(self.test)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    def test_integration2(self):
        """Class method test_integration1 test if MissForest can run properly
        when its 'fit' and 'transform' methods are called."""

        mf = MissForest()
        train_imputed = mf.fit_transform(self.train)
        test_imputed = mf.transform(self.test)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    def test_integration3(self):
        """Class method test_integration1 test if MissForest can run properly
        when its 'fit' and 'transform' methods are called."""

        mf = MissForest()
        mf.fit(self.train)
        train_imputed = mf.fit_transform(self.train)
        test_imputed = mf.fit_transform(self.test)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    def test_integration4(self):
        """Class method test_integration1 test if MissForest can run properly
        when its 'fit' and 'transform' methods are called."""

        mf = MissForest()
        mf.fit(self.train)
        test_imputed = mf.fit_transform(self.test)
        train_imputed = mf.fit_transform(self.train)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
