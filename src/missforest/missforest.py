"""This module contains `MissForest` code."""

from ._info import VERSION, AUTHOR

__all__ = ["MissForest"]
__version__ = VERSION
__author__ = AUTHOR

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
from ._metrics import pfc, nrmse
from typing import Any, Tuple, Iterable, Dict
from sklearn.base import BaseEstimator
from tqdm import tqdm
import warnings


lgbm_clf = LGBMClassifier(verbosity=-1)
lgbm_rgr = LGBMRegressor(verbosity=-1)


class MissForest:
    """
    Attributes
    ----------
    classifier : Union[Any, BaseEstimator]
        Estimator that predicts missing values of categorical columns.
    regressor : Union[Any, BaseEstimator]
        Estimator that predicts missing values of numerical columns.
    initial_guess : str
        Determines the method of initial imputation.
    max_iter : int
        Maximum iterations of imputing.
    early_stopping : bool
        Determines if early stopping will be executed.
    categorical_columns : list
        All categorical columns of given dataframe `x`.
    numerical_columns : list
        All numerical columns of given dataframe `x`.
    _is_fitted : bool
        A state that determines if an instance of `MissForest` is fitted.

    Methods
    -------
    _get_missing_rows(x: pd.DataFrame)
        Gather the indices of any rows that have missing values.
    _get_map_and_rev_map(self, x: pd.DataFrame)
        Gets the encodings and the reverse encodings of categorical variables.
    _compute_initial_imputations(self, x: pd.DataFrame,
                                     categorical: Iterable[Any])
        Computes and stores the initial imputation values for each feature
        in `x`.
    _initial_impute(x: pd.DataFrame,
                        initial_imputations: Dict[Any, Union[str, np.float64]])
        Imputes the values of features using the mean or median for
        numerical variables; otherwise, uses the mode for imputation.
    _add_unseen_categories(x, mappings)
        Updates mappings and reverse mappings based on any unseen categories
        encountered.
    fit(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
        Checks if the arguments are valid and initializes different class
        attributes.
    transform(self, x: pd.DataFrame)
        Imputes all missing values in `x`.
    fit_transform(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
        Calls class methods `fit` and `transform` on `x`.
    """

    def __init__(self, clf: Union[Any, BaseEstimator] = lgbm_clf,
                 rgr: Union[Any, BaseEstimator] = lgbm_rgr,
                 initial_guess: str = "median", max_iter: int = 5,
                 early_stopping=True) -> None:
        """
        Parameters
        ----------
        clf : estimator object, default=None.
            This object is assumed to implement the scikit-learn estimator api.
        rgr : estimator object, default=None.
            This object is assumed to implement the scikit-learn estimator api.
        max_iter : int, default=5
            Determines the number of iteration.
        initial_guess : str, default=`median`
            If `mean`, initial imputation will be the mean of the features.
            If `median`, initial imputation will be the median of the features.
        early_stopping : bool
            Determines if early stopping will be executed.

        Raises
        ------
        ValueError
            - If argument `clf` is not an estimator.
            - If argument `rgr` is not an estimator.
            - If argument `initial_guess` is not a str.
            - If argument `initial_guess` is neither `mean` nor `median`.
            - If argument `max_iter` is not an int.
            - If argument `early_stopping` is not a bool.
        """
        if not _is_estimator(clf):
            raise ValueError("Argument `clf` only accept estimators that has "
                             "class methods `fit` and `predict`.")

        if not _is_estimator(rgr):
            raise ValueError("Argument `rgr` only accept estimators that has "
                             "class methods `fit` and `predict`.")

        if not isinstance(initial_guess, str):
            raise ValueError("Argument `initial_guess` must be str.")

        if initial_guess not in ("median", "mean"):
            raise ValueError("Argument `initial_guess` can only be `median` "
                             "or `mean`.")

        if not isinstance(max_iter, int):
            raise ValueError("Argument `max_iter` must be int.")

        if not isinstance(early_stopping, bool):
            raise ValueError("Argument `early_stopping` must be bool.")

        self.classifier = clf
        self.regressor = rgr
        self.initial_guess = initial_guess
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.categorical_columns = None
        self.numerical_columns = None
        self._is_fitted = False

    @staticmethod
    def _get_missing_rows(x: pd.DataFrame) -> Dict[Any, pd.Index]:
        """Gather the indices of any rows that have missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        missing_rows : dict
            Dictionary containing features with missing values as keys,
            and their corresponding indices as values.
        """
        missing_row = {}
        for c in x.columns:
            feature = x[c]
            is_missing = feature.isnull()
            missing_index = feature[is_missing].index
            if len(missing_index) > 0:
                missing_row[c] = missing_index

        return missing_row

    def _get_map_and_rev_map(
            self, x: pd.DataFrame
    ) -> Union[Tuple[Dict[Any, int], Dict[int, Any]], Tuple[Dict, Dict]]:
        """Gets the encodings and the reverse encodings of categorical
        variables.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be encoded.

        Returns
        -------
        mappings : dict
            Dictionary containing the categorical variables as keys and
            their corresponding encodings as values.
        rev_mappings : dict
            Dictionary containing the categorical variables as keys and
            their corresponding reverse encodings as values.
        """
        mappings = {}
        rev_mappings = {}

        for c in x.columns:
            if c in self.categorical_columns:
                unique = x[c].dropna().unique()
                n_unique = range(x[c].dropna().nunique())

                mappings[c] = dict(zip(unique, n_unique))
                rev_mappings[c] = dict(zip(n_unique, unique))

        return mappings, rev_mappings

    def _compute_initial_imputations(self, x: pd.DataFrame,
                                     categorical: Iterable[Any]
                                     ) -> Dict[Any, Union[str, np.float64]]:
        """Computes and stores the initial imputation values for each feature
        in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            The dataset consisting solely of features that require imputation.
        categorical : Iterable[Any]
            An iterable containing identifiers for all categorical features
            present in `x`.

        Raises
        ------
        ValueError
            - Raised if any feature specified in the `categorical` argument
            does not exist within the columns of `x`.
            - Raised if the `initial_guess` argument is provided and its
            value is neither 'mean' nor 'median'.
        """
        intersection = set(categorical).intersection(set(x.columns))
        if not intersection == set(categorical):
            raise ValueError("Not all features in argument `categorical` "
                             "existed in `x` columns.")

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
                    raise ValueError("Argument `initial_guess` only accepts "
                                     "`mean` or `median`.")

        return initial_imputations

    @staticmethod
    def _initial_impute(x: pd.DataFrame,
                        initial_imputations: Dict[Any, Union[str, np.float64]]
                        ) -> pd.DataFrame:
        """Imputes the values of features using the mean or median for
        numerical variables; otherwise, uses the mode for imputation.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.
        initial_imputations : dict
            Dictionary containing initial imputation values for each feature.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """
        for c in x.columns:
            x[c] = x[c].fillna(initial_imputations[c])

        return x

    @staticmethod
    def _add_unseen_categories(
            x, mappings
    ) -> Union[Tuple[Dict[Any, int], Dict[int, Any]], Tuple[Dict, Dict]]:
        """Updates mappings and reverse mappings based on any unseen
        categories encountered.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            The dataset consisting solely of features that require imputation.
        mappings : dict
            A dictionary mapping categorical variables to their encoded
            representations.

        Returns
        -------
        rev_mappings : dict
            A dictionary mapping categorical variables to their original
            values, effectively serving as the reverse of the `mappings`
            parameter.
        updated_mappings : dict
            An updated dictionary reflecting the latest mappings between
            categorical variables and their encoded representations,
            incorporating any new categories encountered during processing.
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

    def _is_stopping_criterion_satisfied(self, pfc_score: list[float],
                                         nrmse_score: list[float]) -> bool:
        """Checks if stopping criterion satisfied. If satisfied, return True.
        Otherwise, return False.

        Parameters
        ----------
        pfc_score : list[float]
            Latest 2 PFC scores.
        nrmse_score : list[float]
            Latest 2 NRMSE scores.

        Returns
        -------
        bool
            True, if stopping criterion satisfied.
            False, if stopping criterion not satisfied.
        """
        is_pfc_increased = False
        if any(self.categorical_columns) and len(pfc_score) >= 2:
            is_pfc_increased = pfc_score[-1] > pfc_score[-2]

        is_nrmse_increased = False
        if any(self.numerical_columns) and len(nrmse_score) >= 2:
            is_nrmse_increased = nrmse_score[-1] > nrmse_score[-2]

        if (
                any(self.categorical_columns) and
                any(self.numerical_columns) and
                is_pfc_increased * is_nrmse_increased
        ):
            warnings.warn("Both PFC and NRMSE have increased.")
            return True
        elif (
                any(self.categorical_columns) and
                not any(self.numerical_columns) and
                is_pfc_increased
        ):
            warnings.warn("PFC have increased.")
            return True
        elif (
                not any(self.categorical_columns) and
                any(self.numerical_columns) and
                is_nrmse_increased
        ):
            warnings.warn("NRMSE increased.")
            return True

        return False

    def fit(self, x: pd.DataFrame, categorical: Iterable[Any] = None):
        """Checks if the arguments are valid and initializes different class
        attributes.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.
        categorical : Iterable[Any], default=None
            All categorical features of x.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Reverse label-encoded dataset (features only).

        Raises
        ------
        ValueError
            - If argument `x` is not a pandas DataFrame, NumPy array, or
              list of lists.
            - If argument `categorical` is not a list of strings or NoneType.
            - If argument `categorical` is NoneType and has a length of
              less than one.
            - If there are inf values present in argument `x`.
            - If there are one or more columns with all rows missing.
        """
        x = x.copy()

        # Make sure `x` is either pandas dataframe, numpy array or list of
        # lists.
        if (
                not isinstance(x, pd.DataFrame) and
                not isinstance(x, np.ndarray) and
                not (
                        isinstance(x, list) and
                        all(isinstance(i, list) for i in x)
                )
        ):
            raise ValueError("Argument `x` can only be pandas dataframe, "
                             "numpy array or list of list.")

        # If `x` is a list of list, convert `x` into a pandas dataframe.
        if (
                isinstance(x, np.ndarray) or
                (isinstance(x, list) and all(isinstance(i, list) for i in x))
        ):
            x = pd.DataFrame(x)

        # Make sure `categorical` is a list of str.
        if (
                categorical is not None and
                not isinstance(categorical, list) and
                not all(isinstance(elem, str) for elem in categorical)
        ):
            raise ValueError("Argument `categorical` can only be list of "
                             "str or NoneType.")

        # Make sure `categorical` has at least one variable in it.
        if categorical is not None and len(categorical) < 1:
            raise ValueError(f"Argument `categorical` has a len of "
                             f"{len(categorical)}.")

        # Check for positive or negative inf.
        if (
                categorical is not None and
                np.any(np.isinf(x.drop(categorical, axis=1)))
        ):
            raise ValueError("+/- inf values are not supported.")

        # Make sure there is no column with all missing values.
        if np.any(x.isnull().sum() == len(x)):
            raise ValueError("One or more columns have all rows missing.")

        _validate_single_datatype_features(x)

        if categorical is None:
            categorical = []

        self.categorical_columns = categorical
        self.numerical_columns = [c for c in x.columns if c not in categorical]
        self._is_fitted = True

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Imputes all missing values in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        pd.DataFrame
            - Before last imputation matrix, if stopping criterion is
              triggered.
            - Last imputation matrix, if all iterations are done.

        Raises
        ------
        NotFittedError
            If `MissForest` is not fitted.
        ValueError
            If there are no missing values in `x`.
        """
        x = x.copy()
        if x.isnull().sum().sum() == 0:
            raise ValueError("Argument `x` must contains at least one "
                             "missing value.")

        if not self._is_fitted:
            raise NotFittedError("MissForest is not fitted yet.")

        missing_rows = self._get_missing_rows(x)
        initial_imputations = self._compute_initial_imputations(
            x, self.categorical_columns)
        x_imp = self._initial_impute(x, initial_imputations)
        mappings, rev_mappings = self._get_map_and_rev_map(x)
        mappings, rev_mappings = self._add_unseen_categories(x_imp, mappings)
        x_imp = _label_encoding(x_imp, mappings)

        x_imps = []
        x_imp_cat = []
        x_imp_num = []
        pfc_score = []
        nrmse_score = []
        for _ in tqdm(range(self.max_iter)):
            for c in missing_rows:
                if c in mappings:
                    estimator = deepcopy(self.classifier)
                else:
                    estimator = deepcopy(self.regressor)

                # Fit estimator with imputed x.
                x_obs = x_imp.drop(c, axis=1)
                y_obs = x_imp[c]
                estimator.fit(x_obs, y_obs)

                # Predict the missing column with the trained estimator.
                miss_index = missing_rows[c]
                x_missing = x_imp.loc[miss_index]
                x_missing = x_missing.drop(c, axis=1)
                y_pred = estimator.predict(x_missing)
                y_pred = pd.Series(y_pred)
                y_pred.index = missing_rows[c]

                # Update imputed matrix.
                x_imp.loc[miss_index, c] = y_pred

            # Make sure the sizes of `x_imp_cat`, `x_imp_num` and `x_imps`
            # never grow more than 2 elements.
            if len(x_imp_cat) >= 2:
                x_imp_cat.pop(0)

            if len(x_imp_num) >= 2:
                x_imp_num.pop(0)

            if len(x_imps) >= 2:
                x_imps.pop(0)

            # Make sure the sizes of `pfc_score` and `nrmse_score` never grow
            # more than 2 elements.
            if len(pfc_score) >= 2:
                pfc_score.pop(0)

            if len(nrmse_score) >= 2:
                nrmse_score.pop(0)

            # Store imputed categorical and numerical features after
            # each iteration.
            x_imp_cat.append(
                x_imp[self.categorical_columns].reset_index(drop=True))
            x_imp_num.append(
                x_imp[self.numerical_columns].reset_index(drop=True))
            x_imps.append(x_imp)

            # Compute and store PFC.
            if any(self.categorical_columns) and len(x_imp_cat) >= 2:
                pfc_score.append(pfc(x_imp_cat[-1], x_imp_cat[-2]))
            
            # Compute and store NRMSE.
            if any(self.numerical_columns) and len(x_imp_num) >= 2:
                nrmse_score.append(nrmse(x_imp_num[-1], x_imp_num[-2]))

            if self._is_stopping_criterion_satisfied(pfc_score, nrmse_score):
                warnings.warn("Stopping criterion triggered. Before last "
                              "imputation matrix will be returned.")
                return _rev_label_encoding(x_imps[-2], rev_mappings)

        # Mapping encoded values back to its categories.
        return _rev_label_encoding(x_imps[-1], rev_mappings)

    def fit_transform(self, x: pd.DataFrame, categorical: Iterable[Any] = None
                      ) -> pd.DataFrame:
        """Calls class methods `fit` and `transform` on `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.
        categorical : Iterable[Any], default=None
            All categorical features of `x`.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """
        self.fit(x, categorical)

        return self.transform(x)
