import pandas as pd
import numpy as np


class MissForest:
    """
    Parameters
    ----------
    classifier : estimator object
       A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    regressor : estimator object
       A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    n_iter : int, default=5
       Argument that determines the how many times MissForest to iterate."""

    def __init__(self, classifier, regressor, n_iter=5):
        self.classifier = classifier
        self.regressor = regressor
        self.n_iter = n_iter

    def fit(self, x):
        """Parameters
           ----------

            X : {array-like, sparse matrix} of shape (n_samples, n_features)
               Training data."""

        def _single_impute(self, x, feature_to_be_imputed, estimator):
            """
           Parameters
           ----------

           x : {array-like, sparse matrix} of shape (n_samples, n_features)
               Dataset to be imputed.

           estimator : estimator object implementing ‘fit’
               A object of that type is instantiated for each search point.
               This object is assumed to implement the scikit-learn estimator api.
               Either estimator needs to provide a ``score`` function,
               or ``scoring`` must be passed."""
            # Disable pandas warning
            pd.options.mode.chained_assignment = None
            x2 = x.copy()

            cols_to_be_encoded = [column for column in x.columns if isinstance(x[column].sample(1).values[0], str)]
            # If feature in x is string, then label encoding the feature
            for column in cols_to_be_encoded:
                non_na_categories = x[column].dropna().unique()
                labels = range(x[column].dropna().nunique())
                x2[column].replace(non_na_categories, labels, inplace=True)

                if column == feature_to_be_imputed:
                    impute_map = {label: category for category, label in zip(non_na_categories, labels)}

            non_null_features = list(
                set([feat for feat, val in zip(x.isnull().sum().index, x.isnull().sum()) if val == 0] + [
                    feature_to_be_imputed]))

            x2 = x2[non_null_features].copy()
            for i in range(self.n_iter):
                x2['Training/Predict'] = np.where(x2[feature_to_be_imputed].isnull(), 'predict', 'training')
                predict = x2[x2['Training/Predict'] == 'predict']
                training = x2[x2['Training/Predict'] == 'training']

                # If it is first iteration
                if i == 1:
                    # The feature is numeric, then impute it with the median
                    if feature_to_be_imputed in cols_to_be_encoded:
                        feature_median = x2[feature_to_be_imputed].median()
                        predict[feature_to_be_imputed].fillna(feature_median, inplace=True)

                    # The feature is string, then impute it with the mode
                    else:
                        feature_mode = x2[feature_to_be_imputed].mode().values[0]
                        predict[feature_to_be_imputed].fillna(feature_mode, inplace=True)

                if i > 1:
                    for idx, pred in zip(predict.index, y_miss_pred):
                        predict.loc[idx, feature_to_be_imputed] = pred

                # Train the estimator
                X_train = training[non_null_features].drop(feature_to_be_imputed, axis=1)
                y_train = training[feature_to_be_imputed]
                estimator.fit(X_train, y_train)

                if predict[non_null_features].drop(feature_to_be_imputed, axis=1).shape[0] != 0:
                    y_miss_pred = estimator.predict(predict[non_null_features].drop(feature_to_be_imputed, axis=1))

                else:
                    raise Exception(f'No missing values in "{feature_to_be_imputed}".')

                i += 1

            pd.options.mode.chained_assignment = 'warn'

            # Combine the predictions with the non-imputed values and sort them according to the original index
            imputed_feature = pd.concat([training[feature_to_be_imputed], predict[feature_to_be_imputed]],
                                        axis=0).sort_index()

            if feature_to_be_imputed in cols_to_be_encoded:
                imputed_feature.replace(impute_map, inplace=True)

            return imputed_feature


        x = x.copy()
        imputed_x = x.copy()
        features_that_require_imputation_in_ascending_order = x.isnull().sum()[x.isnull().sum() != 0].sort_values().index

        # Check which feature is string
        is_string = {}
        for feature in features_that_require_imputation_in_ascending_order:
            feature_sample = x[~x[feature].isnull()].loc[:, feature].sample(1).values[0]
            is_string[feature] = isinstance(feature_sample, str)

        # If feature is string, impute it with a classifier.
        # If feature is numeric, impute it with a regressor
        for feature in features_that_require_imputation_in_ascending_order:
            if is_string[feature]:
                imputed_feature = _single_impute(x, feature, self.classifier, self.n_iter)

            else:
                imputed_feature = _single_impute(x, feature, self.regressor, self.n_iter)

            # Drop original features and Insert imputed features using the correct indexes
            insert_index = list(x.columns).index(feature)
            imputed_x.drop(feature, axis=1, inplace=True)
            imputed_x.insert(insert_index, feature, imputed_feature)

        return imputed_x
