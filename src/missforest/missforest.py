"""This module contains 'MissForest' code."""

__all__ = ["MissForest"]
__version__ = "2.5.5"
__author__ = "Yuen Shing Yan Hindy"

from copy import deepcopy
from typing import Union
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from ._errors import NotFittedError
from ._validate import (
    _is_estimator,
    _validate_single_datatype_features,
)
from ._label_encoding import (
    _label_encoding,
    _rev_label_encoding
)
from typing import Any, Tuple, Iterable, Dict
from sklearn.base import BaseEstimator
import warnings


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
        If `mean`, initial imputation will use the median of the features.
        If `median`, initial imputation will use the median of the features.
    """

    def __init__(self, clf: Union[Any, BaseEstimator] = LGBMClassifier(),
                 rgr: Union[Any, BaseEstimator] = LGBMRegressor(),
                 initial_guess: str = "median", max_iter: int = 5) -> None:
        # make sure the classifier is None (no input) or an estimator.
        if not _is_estimator(clf):
            raise ValueError("Argument 'clf' only accept estimators that has "
                             "class methods 'fit' and 'predict'.")

        # make sure the regressor is None (no input) or an estimator.
        if not _is_estimator(rgr):
            raise ValueError("Argument 'rgr' only accept estimators that has "
                             "class methods 'fit' and 'predict'.")

        # make sure 'initial_guess' is str.
        if not isinstance(initial_guess, str):
            raise ValueError("Argument 'initial_guess' only accept str.")

        # make sure 'initial_guess' is either 'median' or 'mean'.
        if initial_guess not in ("median", "mean"):
            raise ValueError("Argument 'initial_guess' can only be 'median' "
                             "or 'mean'.")

        # make sure 'max_iter' is int.
        if not isinstance(max_iter, int):
            raise ValueError("Argument 'max_iter' only accept int.")

        self.classifier = clf
        self.regressor = rgr
        self.initial_guess = initial_guess
        self.max_iter = max_iter

        self.categorical = None
        self.numerical = None
        self._is_fitted = False

    @staticmethod
    def _get_missing_rows(x: pd.DataFrame) -> Dict[Any, pd.Index]:
        """
        Gather the index of any rows that has missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        Returns
        -------
        missing_row : dict
            Dictionary that contains features which has missing values as
            keys, and their corresponding indexes as values.
        """

        missing_row = {}
        for c in x.columns:
            feature = x[c]
            is_missing = feature.isnull() > 0
            missing_index = feature[is_missing].index
            if len(missing_index) > 0:
                missing_row[c] = missing_index

        return missing_row

    @staticmethod
    def _get_obs_rows(x: pd.DataFrame) -> pd.Index:
        """
        Gather the rows of any rows that do not have any missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        Returns
        -------
        pd.Index
            Indexes of rows that do not have any missing values.
        """

        n_null = x.isnull().sum(axis=1)

        return x[n_null == 0].index

    def _get_map_and_rev_map(
            self, x: pd.DataFrame
    ) -> Union[Tuple[Dict[Any, int], Dict[int, Any]], Tuple[Dict, Dict]]:
        """
        Gets the encodings and the reverse encodings of categorical variables.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        Returns
        -------
        mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.

        rev_mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.
        """

        mappings = {}
        rev_mappings = {}

        for c in x.columns:
            if c in self.categorical:
                unique = x[c].dropna().unique()
                n_unique = range(x[c].dropna().nunique())

                mappings[c] = dict(zip(unique, n_unique))
                rev_mappings[c] = dict(zip(n_unique, unique))

        return mappings, rev_mappings

    def _compute_initial_imputations(self, x: pd.DataFrame,
                                     categorical: Iterable[Any]
                                     ) -> Dict[Any, Union[str, np.float64]]:
        """
        Computes and stores the initial imputation values of each features
        in x.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        categorical: Iterable[Any]
            All categorical features of x.

        Raises
        ------
        ValueError
            If not all features in argument 'categorical' existed in 'x'
            columns.
            If argument 'initial_guess' is not 'mean' or 'median'.
        """

        intersection = set(categorical).intersection(set(x.columns))
        if not intersection == set(categorical):
            raise ValueError("Not all features in argument 'categorical' "
                             "existed in 'x' columns.")

        initial_imputations = {}
        for c in x.columns:
            if c in categorical:
                initial_imputations[c] = x[c].mode().values[0]
            else:
                if self.initial_guess == "mean":
                    initial_imputations[c] = x[c].mean()
                elif self.initial_guess == "median":
                    initial_imputations[c] = x[c].median()
                else:
                    raise ValueError("Argument 'initial_guess' only accepts "
                                     "'mean' or 'median'.")

        return initial_imputations

    @staticmethod
    def _initial_impute(x: pd.DataFrame,
                        initial_imputations: Dict[Any, Union[str, np.float64]]
                        ) -> pd.DataFrame:
        """
        Imputes the values of features using the mean or median if they are
        numerical variables, else, imputes with mode.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        initial_imputations : dict
            All initial imputation values for each feature.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Imputed Dataset (features only).
        """

        for c in x.columns:
            x[c].fillna(initial_imputations[c], inplace=True)

        return x

    @staticmethod
    def _add_unseen_categories(
            x, mappings
    ) -> Union[Tuple[Dict[Any, int], Dict[int, Any]], Tuple[Dict, Dict]]:
        """
        Updates mappings and reverse mappings, if there are any unseen
        categories.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.

        Returns
        -------
        rev_mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.

        mappings : dict
            Dictionary that contains the categorical variables as keys and
            their corresponding encodings as values.
        """

        for k, v in mappings.items():
            for category in x[k].unique():
                if category not in v:
                    warnings.warn("Unseen category found in dataset. "
                                  "New label will be added.")
                    mappings[k][category] = max(v.values()) + 1

        rev_mappings = {
            k: {v2: k2 for k2, v2 in v.items()} for k, v in mappings.items()}

        return mappings, rev_mappings

    def compute_gamma_categorical(self, all_x_imp_cat: list
                                  ) -> Union[np.ndarray, int]:
        """
        Compute and returns Gamma of categorical variables in imputed 'x'.

        Parameters
        ----------
        all_x_imp_cat: list
            All categorical variables in imputed 'x'.

        Returns
        -------
        np.ndarray
            Gamma of categorical variables in imputed 'x'.
        int
            Gamma of 0, to indicates that there is no differences in
            imputed 'x'.
        """

        if len(self.categorical) > 0 and len(all_x_imp_cat) >= 2:
            x_imp_cat = all_x_imp_cat[-1]
            x_imp_cat_prev = all_x_imp_cat[-2]
            return (np.sum(np.sum(x_imp_cat != x_imp_cat_prev, axis=0),
                           axis=0) / len(self.categorical))
        else:
            return 0

    def compute_gamma_numerical(self, all_x_imp_num: list
                                ) -> Union[np.ndarray, int]:
        """
        Compute and returns Gamma of numerical variables in imputed 'x'.

        Parameters
        ----------
        all_x_imp_num: list
            All numerical variables in imputed 'x'.

        Returns
        -------
        np.ndarray
            Gamma of numerical variables in imputed 'x'.
        int
            Gamma of 0, to indicates that there is no differences in
            imputed 'x'.
        """

        if len(self.numerical) > 0 and len(all_x_imp_num) >= 2:
            x_imp_num = all_x_imp_num[-1]
            x_imp_num_prev = all_x_imp_num[-2]
            return np.sum(np.sum(
                (x_imp_num - x_imp_num_prev) ** 2, axis=0
            ), axis=0) / np.sum(np.sum(x_imp_num ** 2, axis=0), axis=0)
        else:
            return 0

    def fit(self, x: pd.DataFrame, categorical: Iterable[Any] = None):
        """
        Checks if the arguments are valid and initiates different class
        attributes.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        categorical: Iterable[Any], default=None
            All categorical features of x.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Reverse label-encoded dataset (features only).

        Raises
        ------
        ValueError
            - If argument 'x' is not pandas dataframe, numpy array or list
              of list.
            - If argument 'categorical' is not list of str or NoneType.
            - If argument 'categorical' is NoneType, and it has length of less
              than one.
            - If there inf values presents in argument 'x'.
            - If there are one or more columns have all rows missing.
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
            raise ValueError("Argument 'x' can only be pandas dataframe, "
                             "numpy array or list of list.")

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

        _validate_single_datatype_features(x)

        if categorical is None:
            categorical = []

        self.categorical = categorical
        self.numerical = [c for c in x.columns if c not in categorical]
        self._is_fitted = True

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes all missing values in 'x'.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        Returns
        -------
        Imputed dataset (features only).

        Raises
        ------
        NotFittedError
            If 'MissForest' is not fitted.

        ValueError
            If there are no missing values in 'x'.
        """

        x = x.copy()
        if x.isnull().sum().sum() == 0:
            raise ValueError("Argument 'x' must contains at least one "
                             "missing value.")

        if not self._is_fitted:
            raise NotFittedError("MissForest is not fitted yet.")

        missing_rows = self._get_missing_rows(x)
        obs_rows = self._get_obs_rows(x)
        initial_imputations = self._compute_initial_imputations(
            x, self.categorical)
        x_imp = self._initial_impute(x, initial_imputations)
        mappings, rev_mappings = self._get_map_and_rev_map(x)
        mappings, rev_mappings = self._add_unseen_categories(x_imp, mappings)
        x_imp = _label_encoding(x_imp, mappings)

        all_x_imp_cat = []
        all_x_imp_num = []
        all_gamma_cat = []
        all_gamma_num = []
        n_iter = 0
        while True:
            for c in missing_rows:
                if c in mappings:
                    estimator = deepcopy(self.classifier)
                else:
                    estimator = deepcopy(self.regressor)

                # Fit estimator with imputed x
                x_obs = x_imp.drop(c, axis=1).loc[obs_rows]
                y_obs = x_imp[c].loc[obs_rows]
                estimator.fit(x_obs, y_obs)

                # Predict the missing column with the trained estimator
                miss_index = missing_rows[c]
                x_missing = x_imp.loc[miss_index]
                x_missing = x_missing.drop(c, axis=1)
                y_pred = estimator.predict(x_missing)
                y_pred = pd.Series(y_pred)
                y_pred.index = missing_rows[c]

                # Update imputed matrix
                x_imp.loc[miss_index, c] = y_pred

                all_x_imp_cat.append(
                    x_imp[self.categorical].reset_index(drop=True))
                all_x_imp_num.append(
                    x_imp[self.numerical].reset_index(drop=True))

            all_gamma_cat.append(self.compute_gamma_categorical(all_x_imp_cat))
            all_gamma_num.append(self.compute_gamma_numerical(all_x_imp_num))

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
        return _rev_label_encoding(x_imp, rev_mappings)

    def fit_transform(self, x: pd.DataFrame, categorical: Iterable[Any] = None
                      ) -> pd.DataFrame:
        """
        Calls class method 'fit' and 'transform' on 'x'.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needed to be imputed.

        categorical: Iterable[Any], default=None
            All categorical features of x.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """

        self.fit(x, categorical)

        return self.transform(x)
