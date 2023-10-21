import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from missforest.missforest import MissForest
from missforest.errors import InputInvalidError


class TestMissForest(unittest.TestCase):
    def setUp(self):
        self.missforest = MissForest()

    def test_guarding_logic_initial_guess_str(self):
        MissForest(initial_guess="mean")

    def test_guarding_logic_initial_guess_non_str(self):
        with self.assertRaises(ValueError):
            MissForest(initial_guess=10)

    def test_guarding_logic_initial_guess_not_mean_or_median(self):
        with self.assertRaises(ValueError):
            MissForest(initial_guess="mode")

    def test_guarding_logic_max_iter_int(self):
        MissForest(max_iter=3)

    def test_guarding_logic_max_iter_non_int(self):
        with self.assertRaises(ValueError):
            MissForest(max_iter=3.5)

    def test_is_estimator_or_none(self):
        estimator = MagicMock()

        with (
            patch.object(estimator, 'fit', return_value=None),
            patch.object(estimator, 'predict', return_value=None)
        ):
            self.assertTrue(self.missforest._is_estimator_or_none(estimator))
            self.assertTrue(callable(estimator))

    def test_get_missing_rows(self):
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
        # Create a dataframe with missing values
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
        # Create a DataFrame with some missing values
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
        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        df = self.missforest._check_if_valid(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_check_if_valid_numpy_array(self):
        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        df_numpy_array = self.missforest._check_if_valid(df.to_numpy())
        self.assertIsInstance(df_numpy_array, pd.DataFrame)

    def test_check_if_valid_list_of_lists(self):
        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        df_list_of_lists = self.missforest._check_if_valid(df.values.tolist())
        self.assertIsInstance(df_list_of_lists, pd.DataFrame)

    def test_check_if_valid_str(self):
        df = pd.DataFrame({
            'A': ['a', 'b', None],
            'B': [None, 'b', 'c'],
            'C': ['a', 'b', 'c']
        })

        with self.assertRaises(InputInvalidError):
            self.missforest._check_if_valid(df.to_string())

    def test_initial_imputation_mean(self):
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
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, 2, 3, None]
        })

        self.missforest.initial_guess = 'mode'
        imputed_df = self.missforest._initial_imputation(df)
        self.assertEqual(imputed_df['A'].mode()[0], imputed_df['A'].iloc[2])
        self.assertEqual(imputed_df['B'].mode()[0], imputed_df['B'].iloc[0])
        self.assertEqual(imputed_df['C'].mode()[0], imputed_df['C'].iloc[3])
    
    def test_label_encoding(self):
        # Create a DataFrame with missing values
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
