import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

class MissForest:
    """
    Parameters
    ----------
    classifier : estimator object.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.

    regressor : estimator object.
    A object of that type is instantiated for each search point.
    This object is assumed to implement the scikit-learn estimator api.

     n_iter : int
     Determines the number of iteration.

     initial_guess : string, callable or None, default='median'
     If ``mean``, the initial impuatation will use the median of the features.
     If ``median``, the initial impuatation will use the median of the features.
    """

    def __init__(self, classifier: BaseEstimator=RandomForestClassifier(), regressor: BaseEstimator=RandomForestRegressor(), initial_guess: str='median', n_iter: int=5):
        self.classifier = classifier
        self.regressor = regressor
        self.initial_guess = initial_guess
        self.n_iter = n_iter

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.
        """
        miss_row = {}
        for c in X.columns:
            miss_row[c] = X[c][X[c].isnull() > 0].index
        miss_col = X.columns[X.isnull().sum(axis=0).sort_values() > 0]

        obs_row = X[X.isnull().sum(axis=1) == 0].index

        mappings = {}
        rev_mappings = {}
        for c in X.columns:
            if type(X[c].dropna().sample(n=1).values[0]) == str:
                mappings[c] = {k: v for k, v in zip(X[c].dropna().unique(), range(X[c].dropna().nunique()))}
                rev_mappings[c] = {v: k for k, v in zip(X[c].dropna().unique(), range(X[c].dropna().nunique()))}

        non_impute_cols = [c for c in X.columns if c not in mappings.keys()]

        # 1) Make an initial guess for all missing categorical/numeric values (e.g. mean, mode)
        for c in X.columns:
            # if datatype is numeric, fillna with mean or median
            if X[c].dtype in ['int_', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                              'float_',
                              'float16', 'float32', 'float64']:

                if self.initial_guess == 'mean':
                    X[c].fillna(X[c].mean(), inplace=True)

                else:
                    X[c].fillna(X[c].median(), inplace=True)

            # if datatype is categorical, fillna with mode
            else:
                X[c].fillna(X[c].mode().values[0], inplace=True)

        # Label Encoding
        for c in mappings:
            X[c].replace(mappings[c], inplace=True)
            X[c] = X[c].astype(int)

        iter = 0
        while True:
            for c in miss_col:
                if c in mappings:
                    estimator = self.classifier

                else:
                    estimator = self.regressor

                # Fit estimator with imputed X
                estimator.fit(X.drop(c, axis=1).loc[obs_row], X[c].loc[obs_row])

                # Predict the missing column with the trained estimator
                y_pred = estimator.predict(X.loc[miss_row[c]].drop(c, axis=1))
                y_pred = pd.Series(y_pred)
                y_pred.index = miss_row[c]

                # Update imputed matrix
                X.loc[miss_row[c], c] = y_pred

            # Check if Criteria is met
            if iter >= self.n_iter:
                break

            iter += 1

        # Reverse mapping
        for c in rev_mappings:
            X[c].replace(rev_mappings[c], inplace=True)

        self.X = X

        return X
