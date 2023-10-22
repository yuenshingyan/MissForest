"""This module contains class 'TestMissForest'."""

__all__ = ["MissForest"]
__version__ = "2.1.0"
__author__ = "Yuen Shing Yan Hindy"

import unittest
import pandas as pd
from missforest.missforest import MissForest
from missforest.errors import InputInvalidError


class TestMissForest(unittest.TestCase):
    """Class 'TestMissForest' is used for to unit-tests all class methods in
    'TestMissForest'."""

    def setUp(self):
        """Class method 'setUp' is a special method that is automatically
        called before each test method is executed. It is used to set up
        'MissForest' instance that is shared across multiple tests."""

        self.missforest = MissForest()

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
        result = mf._get_missing_rows(df)
        expected_result = {'A': [2], 'B': [0], 'C': [3]}
        self.assertEqual(result, expected_result)

    def test_get_missing_cols(self):
        """Class method 'test_get_missing_cols' test if '_get_missing_cols' of
         'MissForest' correctly gather the columns of any rows that has missing
          values."""

        data = {
            'A': [1, 2, None, 4, 5],
            'B': [None, 2, 3, 4, 5],
            'C': [1, 2, 3, None, 5],
            'D': [1, 2, 3, 4, 5]
        }
        df = pd.DataFrame(data)

        # Call the _get_missing_cols method
        missing_cols = self.missforest._get_missing_cols(df)

        # Define the expected result
        expected_missing_cols = pd.Index(['B', 'C', 'D'])

        # Assert that the result is as expected
        pd.testing.assert_index_equal(missing_cols, expected_missing_cols)

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
        obs_row = self.missforest._get_obs_row(df)

        # Check that the result is a pd.Index object
        self.assertIsInstance(obs_row, pd.Index)

        # Check that the result contains the indexes of the rows with no
        # missing values
        self.assertEqual(obs_row.tolist(), [0, 1, 4])

    def test_get_map_and_rev_map(self):
        """Class method 'test_get_map_and_rev_map' test if
        'test_get_map_and_rev_map' correctly construct dictionaries for label
         encoding and reverse-label encoding."""

        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        miss_forest = MissForest()
        mappings, rev_mappings = miss_forest._get_map_and_rev_map(df)

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

        self.assertEqual(mappings, expected_mappings)
        self.assertEqual(rev_mappings, expected_rev_mappings)

    def test_check_if_valid_pandas_dataframe(self):
        """Class method 'test_check_if_valid_pandas_dataframe' test if
        '_check_if_valid' of 'MissForest' returns a pandas dataframe if
         it is given a pandas dataframe."""

        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        df = self.missforest._check_if_valid(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_check_if_valid_numpy_array(self):
        """Class method 'test_check_if_valid_numpy_array' test if
        '_check_if_valid' of 'MissForest' returns a pandas dataframe if it is
        given a numpy array."""

        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        df_numpy_array = self.missforest._check_if_valid(df.to_numpy())
        self.assertIsInstance(df_numpy_array, pd.DataFrame)

    def test_check_if_valid_list_of_lists(self):
        """Class method 'test_check_if_valid_list_of_lists' test if
        '_check_if_valid' of 'MissForest' returns a pandas dataframe if it is
         given a list of lists."""

        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        df_list_of_lists = self.missforest._check_if_valid(df.values.tolist())
        self.assertIsInstance(df_list_of_lists, pd.DataFrame)

    def test_check_if_valid_str(self):
        """Class method 'test_check_if_valid_str' test if
        '_check_if_valid' of 'MissForest' will raise 'InputInvalidError' if it
         is given a pandas dataframe in string."""

        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        with self.assertRaises(InputInvalidError):
            self.missforest._check_if_valid(df.to_string())

    def test_initial_imputation_mean(self):
        """Class method 'test_initial_imputation_mean' test if the missing
        values are imputed correctly with the mean values if argument
        'initial_guess' is set to 'mean'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, 2, 3, None]
        })

        self.missforest.initial_guess = 'mean'
        imputed_df = self.missforest._initial_imputation(df)
        self.assertEqual(imputed_df['A'].mean(), imputed_df['A'].iloc[2])
        self.assertEqual(imputed_df['B'].mean(), imputed_df['B'].iloc[0])
        self.assertEqual(imputed_df['C'].mean(), imputed_df['C'].iloc[3])

    def test_initial_imputation_median(self):
        """Class method 'test_initial_imputation_median' test if the missing
        values are imputed correctly with the median values if argument
        'initial_guess' is set to 'median'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, 2, 3, None]
        })

        self.missforest.initial_guess = 'median'
        imputed_df = self.missforest._initial_imputation(df)
        self.assertEqual(imputed_df['A'].median(), imputed_df['A'].iloc[2])
        self.assertEqual(imputed_df['B'].median(), imputed_df['B'].iloc[0])
        self.assertEqual(imputed_df['C'].median(), imputed_df['C'].iloc[3])

    def test_initial_imputation_mode(self):
        """Class method 'test_initial_imputation_mode' test if 'ValueError' is
         raised if values other than 'mean' or 'median' is passed to class
         method '_initial_imputation_mode' of 'MissForest'."""

        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, 2, 3, None]
        })

        self.missforest.initial_guess = 'mode'
        with self.assertRaises(ValueError):
            self.missforest._initial_imputation(df)

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


if __name__ == '__main__':
    unittest.main()
