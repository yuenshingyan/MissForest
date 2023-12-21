"""This module contains Python class 'MissForest'."""

__all__ = ["MissForest"]
__version__ = "2.3.1"
__author__ = "Yuen Shing Yan Hindy"

from copy import deepcopy
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from missforest.errors import MultipleDataTypesError, NotFittedError
from typing import Any
from sklearn.base import BaseEstimator


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

    def __init__(self, clf: Any | BaseEstimator = LGBMClassifier(),
                 rgr: Any | BaseEstimator = LGBMRegressor(),
                 initial_guess: str = 'median', max_iter: int = 5) -> None:
        # make sure the classifier is None (no input) or an estimator.
        if not self._is_estimator(clf):
            raise ValueError("Argument 'clf' only accept estimators that has "
                             "class methods 'fit' and 'predict'.")

        # make sure the regressor is None (no input) or an estimator.
        if not self._is_estimator(rgr):
            raise ValueError("Argument 'rgr' only accept estimators that has"
                             " class methods 'fit' and 'predict'.")

        # make sure 'initial_guess' is str.
        if not isinstance(initial_guess, str):
            raise ValueError("Argument 'initial_guess' only accept str.")

        # make sure 'initial_guess' is either 'median' or 'mean'.
        if initial_guess not in ("median", "mean"):
            raise ValueError("Argument 'initial_guess' can only be 'median' or"
                             " 'mean'.")

        # make sure 'max_iter' is int.
        if not isinstance(max_iter, int):
            raise ValueError("Argument 'max_iter' only accept int.")

        self.classifier = clf
        self.regressor = rgr
        self.initial_guess = initial_guess
        self.max_iter = max_iter
        self._initials = {}
        self._missing_row = {}
        self._obs_row = None
        self._mappings = {}
        self._rev_mappings = {}
        self.categorical = None
        self.numerical = None
        self._all_x_imp_cat = []
        self._all_x_imp_num = []
        self._is_fitted = False

    @staticmethod
    def _is_estimator(estimator: Any | BaseEstimator) -> bool:
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

            # assumes it is an estimator if it has 'fit' and 'predict' methods.
            return is_has_fit_method and is_has_predict_method
        except AttributeError:
            return False

    def _get_missing_rows(self, x: pd.DataFrame) -> None:
        """
        Class method '_get_missing_rows' gather the index of any rows that has
        missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        miss_row : dict
        Dictionary that contains features which has missing values as keys, and
        their corresponding indexes as values.
        """

        for c in x.columns:
            feature = x[c]
            is_missing = feature.isnull() > 0
            missing_index = feature[is_missing].index
            if len(missing_index) > 0:
                self._missing_row[c] = missing_index

    def _get_obs_row(self, x: pd.DataFrame) -> None:
        """
        Class method '_get_obs_row' gather the rows of any rows that do not
        have any missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        None
        """

        n_null = x.isnull().sum(axis=1)
        self._obs_row = x[n_null == 0].index

    def _get_map_and_rev_map(self, x: pd.DataFrame, categorical: list
                             ) -> None:
        """
        Class method '_get_map_and_rev_map' gets the encodings and the reverse
        encodings of categorical variables.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        categorical : list
            All categorical features of x.

        Return
        ------
        None
        """

        for c in x.columns:
            if c in categorical:
                unique = x[c].dropna().unique()
                n_unique = range(x[c].dropna().nunique())

                self._mappings[c] = dict(zip(unique, n_unique))
                self._rev_mappings[c] = dict(zip(n_unique, unique))

    @staticmethod
    def _check_if_all_single_type(x: pd.DataFrame) -> None:
        """
        Class method '_check_if_all_single_type' checks if all values in the
        feature belongs to the same datatype. If not, error
        'MultipleDataTypesError will be raised.'

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.
        """

        vectorized_type = np.vectorize(type)
        for c in x.columns:
            feature_no_na = x[c].dropna()
            all_type = vectorized_type(feature_no_na)
            all_unique_type = pd.unique(all_type)
            n_type = len(all_unique_type)
            if n_type > 1:
                raise MultipleDataTypesError(f"Feature {c} has more than one "
                                             f"datatype.")

    def _get_initials(self, x: pd.DataFrame, categorical: list) -> None:
        """
        Class method '_initial_imputation' calculates and stores the initial
        imputation values of each features in x.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        categorical : list
        All categorical features of x.

        Return
        ------
        None
        """

        intersection = set(categorical).intersection(set(x.columns))
        if not intersection == set(categorical):
            raise ValueError("Not all features in argument 'categorical' "
                             "existed in 'x' columns.")

        for c in x.columns:
            if c in categorical:
                self._initials[c] = x[c].mode().values[0]
            else:
                if self.initial_guess == "mean":
                    self._initials[c] = x[c].mean()
                elif self.initial_guess == "median":
                    self._initials[c] = x[c].median()
                else:
                    raise ValueError("Argument 'initial_guess' only accepts "
                                     "'mean' or 'median'.")

    def _initial_imputation(self, x: pd.DataFrame) -> pd.DataFrame:
        """Class method '_initial_imputation' imputes the values of features
        using the mean or median if they are numerical variables, else, imputes
        with mode.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        Return
        ------
        x : pd.DataFrame of shape (n_samples, n_features)
            Imputed Dataset (features only).
        """

        for c in x.columns:
            x[c].fillna(self._initials[c], inplace=True)

        return x

    @staticmethod
    def _label_encoding(x: pd.DataFrame, mappings: dict) -> pd.DataFrame:
        """
        Class method '_label_encoding' performs label encoding on given
        features and the input mappings.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.

        Return
        ------
        x : pd.DataFrame of shape (n_samples, n_features)
            Label-encoded dataset (features only).
        """

        for c in mappings:
            x[c] = x[c].map(mappings[c]).astype(int)

        return x

    @staticmethod
    def _rev_label_encoding(x: pd.DataFrame, rev_mappings: dict
                            ) -> pd.DataFrame:
        """
        Class method '_rev_label_encoding' performs reverse label encoding on
        given features and the input reverse mappings.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        rev_mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.

        Return
        ------
        x : pd.DataFrame of shape (n_samples, n_features)
            Reverse label-encoded dataset (features only).
        """

        for c in rev_mappings:
            x[c] = x[c].map(rev_mappings[c])

        return x

    def fit(self, x: pd.DataFrame, categorical: list = None):
        """
        Class method 'fit' checks if the arguments are valid and initiates
        different class attributes.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        categorical : list, default=None
        All categorical features of x.

        Return
        ------
        x : pd.DataFrame of shape (n_samples, n_features)
        Reverse label-encoded dataset (features only).
        """

        x = x.copy()

        # make sure 'x' is either pandas dataframe, numpy array or list of
        # lists.
        if (
                not isinstance(x, pd.DataFrame) and
                not isinstance(x, np.ndarray) and
                not (
                        isinstance(x, list) and
                        all(isinstance(i, list) for i in x)
                )
        ):
            raise ValueError("Argument 'x' can only be pandas dataframe, numpy"
                             " array or list of list.")

        # if 'x' is a list of list, convert 'x' into a pandas dataframe.
        if (
                isinstance(x, np.ndarray) or
                (isinstance(x, list) and all(isinstance(i, list) for i in x))
        ):
            x = pd.DataFrame(x)

        # make sure 'categorical' is a list of str.
        if (
                categorical is not None and
                not isinstance(categorical, list) and
                not all(isinstance(elem, str) for elem in categorical)
        ):
            raise ValueError("Argument 'categorical' can only be list of "
                             "str or NoneType.")

        # make sure 'categorical' has at least one variable in it.
        if categorical is not None and len(categorical) < 1:
            raise ValueError(f"Argument 'categorical' has a len of "
                             f"{len(categorical)}.")

        # Check for +/- inf
        if (
                categorical is not None and
                np.any(np.isinf(x.drop(categorical, axis=1)))
        ):
            raise ValueError("+/- inf values are not supported.")

        # make sure there is no column with all missing values.
        if np.any(x.isnull().sum() == len(x)):
            raise ValueError("One or more columns have all rows missing.")

        self._initials = {}
        self._missing_row = {}
        self._obs_row = None
        self._mappings = {}
        self._rev_mappings = {}

        if categorical is None:
            categorical = []

        self.categorical = categorical
        self.numerical = [c for c in x.columns if c not in categorical]

        self._check_if_all_single_type(x)
        self._get_map_and_rev_map(x, categorical)
        self._get_initials(x, categorical)
        self._is_fitted = True

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Class method 'transform' imputes all missing values in 'x'.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        Return
        ------
        x : pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """

        if not self._is_fitted:
            raise NotFittedError("MissForest is not fitted yet.")

        x = x.copy()
        self._get_missing_rows(x)
        self._get_obs_row(x)
        x_imp = self._initial_imputation(x)
        x_imp = self._label_encoding(x_imp, self._mappings)

        all_gamma_cat = []
        all_gamma_num = []
        n_iter = 0
        while True:
            for c in self._missing_row:
                if c in self._mappings:
                    estimator = deepcopy(self.classifier)
                else:
                    estimator = deepcopy(self.regressor)

                # Fit estimator with imputed x
                x_obs = x_imp.drop(c, axis=1).loc[self._obs_row]
                y_obs = x_imp[c].loc[self._obs_row]
                estimator.fit(x_obs, y_obs)

                # Predict the missing column with the trained estimator
                miss_index = self._missing_row[c]
                x_missing = x_imp.loc[miss_index]
                x_missing = x_missing.drop(c, axis=1)
                y_pred = estimator.predict(x_missing)
                y_pred = pd.Series(y_pred)
                y_pred.index = self._missing_row[c]

                # Update imputed matrix
                x_imp.loc[miss_index, c] = y_pred

                self._all_x_imp_cat.append(x_imp[self.categorical])
                self._all_x_imp_num.append(x_imp[self.numerical])

            if len(self.categorical) > 0 and len(self._all_x_imp_cat) >= 2:
                x_imp_cat = self._all_x_imp_cat[-1]
                x_imp_cat_prev = self._all_x_imp_cat[-2]
                gamma_cat = (np.sum(np.sum(
                    x_imp_cat != x_imp_cat_prev, axis=0
                ),  axis=0) / len(self.categorical))
                all_gamma_cat.append(gamma_cat)

            if len(self.numerical) > 0 and len(self._all_x_imp_num) >= 2:
                x_imp_num = self._all_x_imp_num[-1]
                x_imp_num_prev = self._all_x_imp_num[-2]
                gamma_num = (np.sum(np.sum(
                    (x_imp_num - x_imp_num_prev) ** 2, axis=0
                ), axis=0) / np.sum(np.sum(x_imp_num ** 2, axis=0), axis=0))
                all_gamma_num.append(gamma_num)

            n_iter += 1
            if n_iter > self.max_iter:
                break

            if (
                    n_iter >= 2 and
                    len(self.categorical) > 0 and
                    len(all_gamma_cat) >= 2 and
                    all_gamma_cat[-1] > all_gamma_cat[-2]
            ):
                break

            if (
                    n_iter >= 2 and
                    len(self.numerical) > 0 and
                    len(all_gamma_cat) >= 2 and
                    all_gamma_num[-1] > all_gamma_num[-2]
            ):
                break

        # mapping the encoded values back to its categories.
        x = self._rev_label_encoding(x_imp, self._rev_mappings)

        return x

    def fit_transform(self, x: pd.DataFrame, categorical: list = None
                      ) -> pd.DataFrame:
        """
        Class method 'fit_transform' calls class method 'fit' and 'transform'
        on 'x'.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        categorical : list, default=None
        All categorical features of x.

        Return
        ------
        x : pd.DataFrame of shape (n_samples, n_features)
        Imputed dataset (features only).
        """

        self.fit(x, categorical)
        x = self.transform(x)

        return x
