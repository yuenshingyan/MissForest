"""This module contains Python class 'MissForest'."""

__all__ = ["MissForest"]
__version__ = "2.0.0"
__author__ = "Yuen Shing Yan Hindy"

from copy import deepcopy
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from missforest.errors import InputInvalidError, MultipleDataTypesError


class MissForest:
    """
    Parameters
    ----------
    clf : estimator object, default=None.
    This object is assumed to implement the scikit-learn estimator api.

    rgr : estimator object, default=None.
    This object is assumed to implement the scikit-learn estimator api.

     max_iter : int, default=5
     Determines the number of iteration.

     initial_guess : string, callable or None, default='median'
     If ``mean``, the initial imputation will use the median of the features.
     If ``median``, the initial imputation will use the median of the features.
    """

    def __init__(self, clf=LGBMClassifier(), rgr=LGBMRegressor(),
                 initial_guess='median', max_iter=5):
        # make sure the classifier is None (no input) or an estimator.
        if not self._is_estimator(clf):
            raise ValueError("Argument 'clf' only accept estimators that has "
                             "class methods 'fit' and 'predict'.")

        # make sure the regressor is None (no input) or an estimator.
        if not self._is_estimator(rgr):
            raise ValueError("Argument 'rgr' only accept estimators that has"
                             " class methods 'fit' and 'predict'.")

        if not isinstance(initial_guess, str):
            raise ValueError("Argument 'initial_guess' only accept str.")

        if initial_guess not in ("median", "mean"):
            raise ValueError("Argument 'initial_guess' can only be 'median' or"
                             " 'mean'.")

        if not isinstance(max_iter, int):
            raise ValueError("Argument 'max_iter' only accept int.")

        self.classifier = clf
        self.regressor = rgr
        self.initial_guess = initial_guess
        self.max_iter = max_iter
    
    @staticmethod
    def _is_estimator(estimator):
        """
        Class method '_is_estimator_or_none' is used to check if argument
        'estimator' is an object that implement the scikit-learn estimator api.

        Parameters
        ----------
        estimator : estimator object
        This object is assumed to implement the scikit-learn estimator api.

        Return
        ------
        If the argument 'estimator' is None or has class method 'fit' and
        'predict', return True.

        Otherwise, return False
        """

        try:
            # get the class methods 'fit' and 'predict' of the estimator.
            is_has_fit_method = getattr(estimator, "fit")
            is_has_predict_method = getattr(estimator, "predict")

            # check if those class method are callable.
            is_has_fit_method = callable(is_has_fit_method)
            is_has_predict_method = callable(is_has_predict_method)

            # assumes it is an estimator if it has 'fit' and 'predict'
            # methods.
            if is_has_fit_method and is_has_predict_method:
                return True
            else:
                return False
        except AttributeError:
            return False
    
    @staticmethod
    def _get_missing_rows(X):
        """
        Class method '_get_missing_rows' gather the index of any rows that has
        missing values.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        miss_row : dict
        Dictionary that contains features which has missing values as keys, and
        their corresponding indexes as values.
        """

        miss_row = {}
        for c in X.columns:
            feature = X[c]
            is_missing = feature.isnull() > 0
            missing_index = feature[is_missing].index
            miss_row[c] = missing_index

        return miss_row
    
    @staticmethod
    def _get_missing_cols(X):
        """
        Class method '_get_missing_cols' gather the columns of any rows that
        has missing values.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        missing_cols : pandas.Index
        The features that have missing values.
        """

        is_missing = X.isnull().sum(axis=0).sort_values() > 0
        missing_cols = X.columns[is_missing]

        return missing_cols
    
    @staticmethod
    def _get_obs_row(X):
        """
        Class method '_get_obs_row' gather the rows of any rows that do not
        have any missing values.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        obs_row : pandas.Index
        The indexes that do not contain any missing values.
        """

        n_null = X.isnull().sum(axis=1)
        obs_row = X[n_null == 0].index

        return obs_row
    
    @staticmethod
    def _get_map_and_rev_map(X):
        """
        Class method '_get_map_and_rev_map' gets the encodings and the reverse
        encodings of categorical variables.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

        rev_mappings : dict
        Dictionary that contains the categorical variables as values and their
        corresponding encodings as keys.
        """

        # using a vectorized version of 'type' function to speed up
        # computations.
        vectorized_type = np.vectorize(type)

        mappings = {}
        rev_mappings = {}
        for c in X.columns:
            feature_without_na = X[c].dropna()
            feature_without_na_type = vectorized_type(feature_without_na)
            is_all_str = all(feature_without_na_type == str)
            if is_all_str:
                unique_values = X[c].dropna().unique()
                nunique_values = range(X[c].dropna().nunique())

                mappings[c] = dict(zip(unique_values, nunique_values))
                rev_mappings[c] = dict(zip(nunique_values, unique_values))

        return mappings, rev_mappings

    @staticmethod
    def _check_if_valid(X):
        """Class method '_check_if_valid' checks if input argument 'X' is
        either a pandas dataframe, numpy array or list of lists. If 'X' is a
        pandas dataframe, returns 'X'. If 'X' is a numpy array or list of lists
        , 'X' will be converted into pandas dataframe and got returned. If 'X'
         is not either pandas dataframe, numpy array or list of lists,
         'InputInvalidError' will be raised.

         Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

         """

        if isinstance(X, pd.DataFrame):
            return X
        elif (
                isinstance(X, np.ndarray) or
                isinstance(X, list) and all(isinstance(i, list) for i in X)
        ):
            X = pd.DataFrame(X)

            return X
        else:
            raise InputInvalidError("""'InputInvalidError' is raised when the
             input argument 'X' is not either pandas dataframe, numpy array or
              list of lists.""")

    @staticmethod
    def _check_if_all_single_type(X):
        """
        Class method '_check_if_all_single_type' checks if all values in the
        feature belongs to the same datatype. If not, error
        'MultipleDataTypesError will be raised.'

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.
        """

        vectorized_type = np.vectorize(type)
        for c in X.columns:
            feature_no_na = X[c].dropna()
            all_type = vectorized_type(feature_no_na)
            all_unique_type = pd.unique(all_type)
            n_type = len(all_unique_type)
            if n_type > 1:
                raise MultipleDataTypesError(f"Feature {c} has more than one "
                                             f"datatype.")
    
    def _initial_imputation(self, X):
        """
        Class method '_initial_imputation' imputes the values of features using
        the mean or median if they are numerical variables, else, imputes with
        mode.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Imputed dataset (features only).
        """

        for c in X.columns:
            try:
                if self.initial_guess == "mean":
                    impute_values = X[c].mean()
                elif self.initial_guess == "median":
                    impute_values = X[c].median()
                else:
                    raise ValueError("Argument 'initial_guess' only accepts "
                                     "'mean' or 'median'.")
            except TypeError:
                impute_values = X[c].mode().values[0]

            X[c].fillna(impute_values, inplace=True)

        return X
    
    @staticmethod
    def _label_encoding(X, mappings):
        """
        Class method '_label_encoding' performs label encoding on given
        features and the input mappings.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

        Return
        ------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Label-encoded dataset (features only).
        """

        for c in mappings:
            X[c] = X[c].map(mappings[c]).astype(int)

        return X
    
    @staticmethod
    def _rev_label_encoding(X, rev_mappings):
        """
        Class method '_rev_label_encoding' performs reverse label encoding on
        given features and the input reverse mappings.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        rev_mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

        Return
        ------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Reverse label-encoded dataset (features only).
        """

        for c in rev_mappings:
            X[c] = X[c].map(rev_mappings[c])

        return X

    def fit_transform(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Imputed dataset (features only).
        """

        X = self._check_if_valid(X)
        self._check_if_all_single_type(X)
        miss_row = self._get_missing_rows(X)
        miss_col = self._get_missing_cols(X)
        obs_row = self._get_obs_row(X)
        mappings, rev_mappings = self._get_map_and_rev_map(X)
        X_imp = self._initial_imputation(X)
        X_imp = self._label_encoding(X_imp, mappings)

        for _ in range(self.max_iter):
            for c in miss_col:
                if c in mappings:
                    estimator = deepcopy(self.classifier)
                else:
                    estimator = deepcopy(self.regressor)

                # Fit estimator with imputed X
                X_obs = X_imp.drop(c, axis=1).loc[obs_row]
                y_obs = X_imp[c].loc[obs_row]
                estimator.fit(X_obs, y_obs)

                # Predict the missing column with the trained estimator
                miss_index = miss_row[c]
                X_missing = X_imp.loc[miss_index]
                X_missing = X_missing.drop(c, axis=1)
                y_pred = estimator.predict(X_missing)
                y_pred = pd.Series(y_pred)
                y_pred.index = miss_row[c]

                # Update imputed matrix
                X_imp.loc[miss_index, c] = y_pred

        # mapping the encoded values back to its categories.
        X = self._rev_label_encoding(X_imp, rev_mappings)

        return X
