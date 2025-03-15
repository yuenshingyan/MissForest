"""This module contains `MissForest` code."""

from ._info import VERSION, AUTHOR

__all__ = ["MissForest"]
__version__ = VERSION
__author__ = AUTHOR

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Union
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from .errors import NotFittedError
from ._validate import (
    _validate_clf,
    _validate_rgr,
    _validate_initial_guess,
    _validate_max_iter,
    _validate_early_stopping,
    _validate_feature_dtype_consistency,
    _validate_2d,
    _validate_cat_var_consistency,
    _validate_categorical,
    _validate_infinite,
    _validate_empty_feature,
    _validate_imputable,
    _validate_verbose,
    _validate_column_consistency,
)
from .metrics import pfc, nrmse
from ._array import SafeArray
from typing import Any, Iterable, Dict
from sklearn.base import BaseEstimator
from tqdm import tqdm
import warnings

import wandb
from wandb.integration.lightgbm import wandb_callback

## added
from sklearn.model_selection import train_test_split
import lightgbm as lgb
##

logger = logging.getLogger(__name__)
lgbm_clf = LGBMClassifier(verbosity=-1, linear_tree=True)
lgbm_rgr = LGBMRegressor(verbosity=-1, linear_tree=True)


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
    _categorical : list
        All categorical columns of given dataframe `x`.
    _numerical : list
        All numerical columns of given dataframe `x`.
    column_order : pd.Index
        Sorting order of features.
    _is_fitted : bool
        A state that determines if an instance of `MissForest` is fitted.
    _estimators : list
        A ordered dictionary that stores estimators for each feature of each
        iteration.
    _verbose : int
        Determines if messages will be printed out.

    Methods
    -------
    _get_n_missing(x: pd.DataFrame)
        Compute and return the total number of missing values in `x`.
    _get_missing_indices(x: pd.DataFrame)
        Gather the indices of any rows that have missing values.
    _compute_initial_imputations(self, x: pd.DataFrame,
                                     categorical: Iterable[Any])
        Computes and stores the initial imputation values for each feature
        in `x`.
    _initial_impute(x: pd.DataFrame,
                        initial_imputations: Dict[Any, Union[str, np.float64]])
        Imputes the values of features using the mean or median for
        numerical variables; otherwise, uses the mode for imputation.
    fit(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
        Fit `MissForest`.
    transform(self, x: pd.DataFrame)
        Imputes all missing values in `x` with fitted estimators.
    fit_transform(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
        Calls class methods `fit` and `transform` on `x`.
    """

    def __init__(
        self,
        clf: Union[Any, BaseEstimator] = lgbm_clf,
        rgr: Union[Any, BaseEstimator] = lgbm_rgr,
        categorical: Iterable[Any] = None,
        num_cat_classes: dict = None,
        initial_guess: str = "median",
        max_iter: int = 5,
        early_stopping: bool = True,
        verbose: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        clf : estimator object, default=None.
            This object is assumed to implement the scikit-learn estimator api.
        rgr : estimator object, default=None.
            This object is assumed to implement the scikit-learn estimator api.
        categorical : Iterable[Any], default=None
            All categorical features of `x`.
        max_iter : int, default=5
            Determines the number of iteration.
        initial_guess : str, default=`median`
            If `mean`, initial imputation will be the mean of the features.
            If `median`, initial imputation will be the median of the features.
        early_stopping : bool
            Determines if early stopping will be executed.
        verbose : int
            Determines if message will be printed out.

        Raises
        ------
        ValueError
            - If argument `clf` is not an estimator.
            - If argument `rgr` is not an estimator.
            - If argument `categorical` is not a list of strings or NoneType.
            - If argument `categorical` is NoneType and has a length of less
              than one.
            - If argument `initial_guess` is not a str.
            - If argument `initial_guess` is neither `mean` nor `median`.
            - If argument `max_iter` is not an int.
            - If argument `early_stopping` is not a bool.
        """
        _validate_clf(clf)
        _validate_rgr(rgr)
        _validate_categorical(categorical)
        _validate_initial_guess(initial_guess)
        _validate_max_iter(max_iter)
        _validate_early_stopping(early_stopping)
        _validate_verbose(verbose)

        self.classifier = clf
        self.regressor = rgr
        self._categorical = [] if categorical is None else categorical
        self.initial_guess = initial_guess
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self._numerical = None
        self.column_order = None
        self.initial_imputations = None
        self._is_fitted = False
        self._estimators = []
        self._verbose = verbose

        ## Hacky rn -- Added stuff here --
        self._num_cat_classes = {} if categorical is None else num_cat_classes

    @staticmethod
    def _get_n_missing(x: pd.DataFrame) -> int:
        """Compute and return the total number of missing values in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        int
            Total number of missing values in `x`.
        """
        return int(x.isnull().sum().sum())

    @staticmethod
    def _get_missing_indices(x: pd.DataFrame) -> Dict[Any, pd.Index]:
        """Gather the indices of any rows that have missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        missing_indices : dict
            Dictionary containing features with missing values as keys,
            and their corresponding indices as values.
        """
        missing_indices = {}
        for c in x.columns:
            feature = x[c]
            missing_indices[c] = feature[feature.isnull()].index

        return missing_indices

    def _compute_initial_imputations(
        self, x: pd.DataFrame, categorical: Iterable[Any]
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
            - If any feature specified in the `categorical` argument does not
            exist within the columns of `x`.
            - If argument `initial_guess` is provided and its value is
            neither `mean` nor `median`.
        """
        initial_imputations = {}
        for c in x.columns:
            if c in categorical:
                initial_imputations[c] = x[c].mode().values[0]
            elif c not in categorical and self.initial_guess == "mean":
                initial_imputations[c] = x[c].mean()
            elif c not in categorical and self.initial_guess == "median":
                initial_imputations[c] = x[c].median()
            elif c not in categorical:
                raise ValueError(
                    "Argument `initial_guess` only accepts `mean` or `median`."
                )

        return initial_imputations

    @staticmethod
    def _initial_impute(
        x: pd.DataFrame, initial_imputations: Dict[Any, Union[str, np.float64]]
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
        x = x.copy()
        for c in x.columns:
            x[c] = x[c].fillna(initial_imputations[c])

        return x

    def _is_stopping_criterion_satisfied(
        self, pfc_score: SafeArray, nrmse_score: SafeArray
    ) -> bool:
        """Checks if stopping criterion satisfied. If satisfied, return True.
        Otherwise, return False.

        Parameters
        ----------
        pfc_score : SafeArray
            Latest 2 PFC scores.
        nrmse_score : SafeArray
            Latest 2 NRMSE scores.

        Returns
        -------
        bool
            - True, if stopping criterion satisfied.
            - False, if stopping criterion not satisfied.
        """
        is_pfc_increased = False
        if any(self._categorical) and len(pfc_score) >= 2:
            is_pfc_increased = pfc_score[-1] >= pfc_score[-2]  # added equality

        is_nrmse_increased = False
        if any(self._numerical) and len(nrmse_score) >= 2:
            is_nrmse_increased = nrmse_score[-1] >= nrmse_score[-2]  # added equality

        if (
            any(self._categorical)
            and any(self._numerical)
            and is_pfc_increased * is_nrmse_increased
        ):
            if self._verbose >= 2:
                warnings.warn("Both PFC and NRMSE have increased.")

            return True
        elif any(self._categorical) and not any(self._numerical) and is_pfc_increased:
            if self._verbose >= 2:
                warnings.warn("PFC have increased.")

            return True
        elif not any(self._categorical) and any(self._numerical) and is_nrmse_increased:
            if self._verbose >= 2:
                warnings.warn("NRMSE increased.")

            return True

        return False

    def fit(self, x: pd.DataFrame):
        """Fit `MissForest`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Reverse label-encoded dataset (features only).

        Raises
        ------
        ValueError
            - If argument `x` is not a pandas DataFrame or NumPy array.
            - If argument `categorical` is not a list of strings or NoneType.
            - If argument `categorical` is NoneType and has a length of
              less than one.
            - If there are inf values present in argument `x`.
            - If there are one or more columns with all rows missing.
        """
        if self._verbose >= 2:
            warnings.warn(
                "Label encoding is no longer performed by default. "
                "Users will have to perform categorical features "
                "encoding by themselves."
            )

        x = x.copy()

        # Make sure `x` is either pandas dataframe, numpy array or list of
        # lists.
        if not isinstance(x, pd.DataFrame) and not isinstance(x, np.ndarray):
            raise ValueError(
                "Argument `x` can only be pandas dataframe, "
                "numpy array or list of list."
            )

        # If `x` is a list of list, convert `x` into a pandas dataframe.
        if isinstance(x, np.ndarray) or (
            isinstance(x, list) and all(isinstance(i, list) for i in x)
        ):
            x = pd.DataFrame(x)

        _validate_2d(x)
        _validate_empty_feature(x)
        _validate_feature_dtype_consistency(x)
        _validate_imputable(x)
        _validate_cat_var_consistency(x.columns, self._categorical)

        if any(self._categorical):
            _validate_infinite(x.drop(self._categorical, axis=1))
        else:
            _validate_infinite(x)

        self._numerical = [c for c in x.columns if c not in self._categorical]

        # Sort column order according to the amount of missing values
        # starting with the lowest amount.
        pct_missing = x.isnull().sum() / len(x)
        self.column_order = pct_missing.sort_values().index
        x = x[self.column_order].copy()

        n_missing = self._get_n_missing(x[self._categorical])
        missing_indices = self._get_missing_indices(x)
        self.initial_imputations = self._compute_initial_imputations(
            x, self._categorical
        )
        x_imp = self._initial_impute(x, self.initial_imputations)

        x_imp_cat = SafeArray(dtype=pd.DataFrame)
        x_imp_num = SafeArray(dtype=pd.DataFrame)
        pfc_score = SafeArray(dtype=float)
        nrmse_score = SafeArray(dtype=float)

        loop = range(self.max_iter)
        if self._verbose >= 1:
            loop = tqdm(loop)

        for loop_iter in loop:
            fitted_estimators = OrderedDict()

            for c in missing_indices:
                if len(missing_indices[c]) == 0 and loop_iter > 0:
                    continue
                if c in self._categorical:
                    estimator = deepcopy(self.classifier)
                else:
                    estimator = deepcopy(self.regressor)

                # Fit estimator with imputed x.
                x_obs = x_imp.drop(c, axis=1)
                y_obs = x_imp[c]
                # estimator.fit(x_obs, y_obs)

                print(missing_indices[c].shape)

                ## ADDED HERE ##
                if c in self._categorical:
                    n_classes = self._num_cat_classes[c]

                if loop_iter == 0:  # train on only observed data first
                    x_obs = x_obs.loc[~x_obs.index.isin(missing_indices[c])]
                    y_obs = y_obs[~y_obs.index.isin(missing_indices[c])]

                x_train, x_val, y_train, y_val = train_test_split(
                    x_obs, y_obs, test_size=0.1, random_state=42
                )
                lgb_train = lgb.Dataset(x_train, label=y_train)
                lgb_eval = lgb.Dataset(x_val, label=y_val, reference=lgb_train)

                feature_info = ""
                max_depth_debugging = 8  # 8  ## TODO: Change this back to 8 afterwards
                if c in self._categorical and n_classes == 2:
                    feature_info = f"binary: {c}"
                    logger.info(f"binary: {c}")
                    params = {
                        "objective": "binary",
                        "device": "cpu",
                        "boosting": "gbdt",
                        "learning_rate": 0.02,
                        "max_depth": max_depth_debugging,  # 8,
                        "metric": ["binary_error", "binary_logloss"],
                        "verbosity": -1,
                        "bagging_fraction": 0.9,
                        "bagging_freq": 1,
                        "feature_fraction": 0.9,
                    }
                elif c in self._categorical and n_classes > 2:
                    feature_info = f"multiclass: {c}"
                    logger.info(f"multiclass: {c}")
                    params = {
                        "objective": "multiclass",
                        "device": "cpu",
                        "num_class": n_classes,
                        "boosting": "gbdt",
                        "learning_rate": 0.02,
                        "max_depth": max_depth_debugging,  # 8,
                        "metric": ["multi_error", "multi_logloss"],
                        "verbosity": -1,
                        "bagging_fraction": 0.9,
                        "bagging_freq": 1,
                        "feature_fraction": 0.9,
                    }
                else:
                    feature_info = f"regression: {c}"
                    logger.info(f"regression: {c}")
                    params = {
                        "objective": "regression",
                        "device": "cpu",
                        "boosting": "gbdt",
                        "learning_rate": 0.02,
                        "max_depth": max_depth_debugging,  # 8,
                        "metric": ["rmse", "huber"],
                        "verbosity": -1,
                        "bagging_fraction": 0.9,
                        "bagging_freq": 1,
                        "feature_fraction": 0.9,
                    }

                wandb.init(
                    project="census-income-ols-missing-test",
                    config={
                        "model": "lightgbm",
                        "params": params,
                        "feature_info": feature_info,
                    },
                )
                estimator = lgb.train(
                    params,
                    train_set=lgb_train,
                    num_boost_round=5000,  ## TODO: Change this back to 5000 afterwards
                    valid_sets=[lgb_eval],
                    valid_names=["eval"],
                    callbacks=[
                        wandb_callback(),
                        lgb.early_stopping(stopping_rounds=25),
                        lgb.log_evaluation(period=25),
                    ],
                )
                wandb.finish()

                # Predict the missing column with the trained estimator.
                x_missing = x_imp.loc[missing_indices[c]].drop(c, axis=1)
                # print(c, len(missing_indices[c]))
                # print(x_imp.loc[missing_indices[c]].drop(c, axis=1)[:10])
                # print(x_missing.any().any())
                if x_missing.any().any():
                    # # Update imputed matrix.
                    # print(missing_indices[feature], feature)
                    # print(len(missing_indices[feature]))
                    # print(estimator.predict(x_obs, num_iteration=estimator.best_iteration).shape)
                    prediction = estimator.predict(
                        x_missing, num_iteration=estimator.best_iteration
                    )
                    if len(prediction.shape) > 1:  # if multiclass
                        # hacky way of doing it rn
                        print(missing_indices[c])
                        print(c)
                        print(x_imp.loc[missing_indices[c], c])
                        print(np.argmax(prediction, axis=1).shape)
                        print(x_imp.loc[missing_indices[c], c].shape)
                        x_imp.loc[missing_indices[c], c] = np.argmax(
                            prediction, axis=1
                        ).tolist()
                    else:
                        x_imp.loc[missing_indices[c], c] = prediction.tolist()

                    # x_imp.loc[missing_indices[c], c] = (
                    #     # estimator.predict(x_missing).tolist()
                    #     estimator.predict(x_missing, num_iteration=estimator.best_iteration).tolist()
                    # )

                # Store trained estimators.
                fitted_estimators[c] = estimator

            self._estimators.append(fitted_estimators)

            # Store imputed categorical and numerical features after
            # each iteration.
            # Compute and store PFC.
            if any(self._categorical):
                x_imp_cat.append(x_imp[self._categorical])

                if len(x_imp_cat) >= 2:
                    pfc_score.append(
                        pfc(
                            x_true=x_imp_cat[-1],
                            x_imp=x_imp_cat[-2],
                            n_missing=n_missing,
                        )
                    )

            # Compute and store NRMSE.
            if any(self._numerical):
                x_imp_num.append(x_imp[self._numerical])

                if len(x_imp_num) >= 2:
                    nrmse_score.append(
                        nrmse(
                            x_true=x_imp_num[-1],
                            x_imp=x_imp_num[-2],
                        )
                    )

            if self.early_stopping and self._is_stopping_criterion_satisfied(
                pfc_score, nrmse_score
            ):
                self._is_fitted = True
                if self._verbose >= 2:
                    warnings.warn(
                        "Stopping criterion triggered during fitting. "
                        "Before last imputation matrix will be returned."
                    )

                # Remove last iteration of estimators.
                self._estimators = self._estimators[:-1]

                return self

        self._is_fitted = True

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Imputes all missing values in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
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
        if self._verbose >= 2:
            warnings.warn(
                "Label encoding is no longer performed by default. "
                "Users will have to perform categorical features "
                "encoding by themselves."
            )

            warnings.warn(
                f"In version {VERSION}, estimator fitting process "
                f"is moved to `fit` method. `MissForest` will now "
                f"imputes unseen missing values with fitted "
                f"estimators with `transform` method. To retain the "
                f"old behaviour, use `fit_transform` to fit the "
                f"whole unseen data instead."
            )

        if not self._is_fitted:
            raise NotFittedError("MissForest is not fitted yet.")

        _validate_2d(x)
        # _validate_empty_feature(x)
        _validate_feature_dtype_consistency(x)
        _validate_imputable(x)
        _validate_cat_var_consistency(x.columns, self._categorical)
        _validate_column_consistency(set(x.columns), set(self.column_order))

        x = x[self.column_order].copy()

        n_missing = self._get_n_missing(x[self._categorical])
        missing_indices = self._get_missing_indices(x)
        x_imp = self._initial_impute(x, self.initial_imputations)

        x_imps = SafeArray(dtype=pd.DataFrame)
        x_imp_cat = SafeArray(dtype=pd.DataFrame)
        x_imp_num = SafeArray(dtype=pd.DataFrame)
        pfc_score = SafeArray(dtype=float)
        nrmse_score = SafeArray(dtype=float)

        loop = range(len(self._estimators))
        if self._verbose >= 1:
            loop = tqdm(loop)

        for i in loop:
            for feature, estimator in self._estimators[i].items():
                if x[feature].isnull().any():
                    print(f"imputing {feature}")
                    x_obs = x_imp.loc[missing_indices[feature]].drop(feature, axis=1)

                    prediction = estimator.predict(
                        x_obs, num_iteration=estimator.best_iteration
                    )
                    if len(prediction.shape) > 1:  # if multiclass
                        # hacky way of doing it rn
                        x_imp.loc[missing_indices[feature], feature] = np.argmax(
                            prediction, axis=1
                        ).tolist()
                    else:
                        x_imp.loc[missing_indices[feature], feature] = (
                            # estimator.predict(x_obs).tolist()
                            estimator.predict(
                                x_obs, num_iteration=estimator.best_iteration
                            ).tolist()
                        )

            # Store imputed categorical and numerical features after
            # each iteration.
            if any(self._categorical):
                x_imp_cat.append(x_imp[self._categorical])

                # Compute and store PFC.
                if len(x_imp_cat) >= 2:
                    pfc_score.append(
                        pfc(
                            x_true=x_imp_cat[-1],
                            x_imp=x_imp_cat[-2],
                            n_missing=n_missing,
                        )
                    )

            if any(self._numerical):
                x_imp_num.append(x_imp[self._numerical])

                # Compute and store NRMSE.
                if len(x_imp_num) >= 2:
                    nrmse_score.append(
                        nrmse(
                            x_true=x_imp_num[-1],
                            x_imp=x_imp_num[-2],
                        )
                    )

            x_imps.append(x_imp)

            if self.early_stopping and self._is_stopping_criterion_satisfied(
                pfc_score, nrmse_score
            ):
                if self._verbose >= 2:
                    warnings.warn(
                        "Stopping criterion triggered during transform. "
                        "Before last imputation matrix will be returned."
                    )

                return x_imps[-2]

        return x_imps[-1]

    def fit_transform(self, x: pd.DataFrame = None) -> pd.DataFrame:
        """Calls class methods `fit` and `transform` on `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """
        return self.fit(x).transform(x)
