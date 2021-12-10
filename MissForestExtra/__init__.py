#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

class MissForestExtra:
    def __init__(self):     
        self.__author__ = 'Yuen Shing Yan Hindy'
        self.__license__= 'MIT License'
        self.__contact__ = 'https://github.com/HindyDS/MissForestExtra'
    
    def single_impute(self, x, feature_to_be_imputed, estimator, max_iter=5):
        # Disable pandas warning
        pd.options.mode.chained_assignment = None
        x2 = x.copy()    
                  
        columns_needed_to_be_encoded = [column for column in x.columns if isinstance(x[column].sample(1).values[0], str)]
        # If feature in x is string, then label encoding the feature
        for column in columns_needed_to_be_encoded:
          non_na_categories = x[column].dropna().unique()
          labels = range(x[column].dropna().nunique())
          x2[column].replace(non_na_categories, labels, inplace=True)

          if column == feature_to_be_imputed:
              impute_map = {label:category for category, label in zip(non_na_categories, labels)}
       
        non_null_features = list(set([feat for feat, val in zip(x.isnull().sum().index, x.isnull().sum()) if val == 0] + [feature_to_be_imputed]))
            
        x2 = x2[non_null_features].copy()
        for i in range(max_iter):
            x2['Training/Predict'] = np.where(x2[feature_to_be_imputed].isnull(), 'predict', 'training')
            predict = x2[x2['Training/Predict'] == 'predict']
            training = x2[x2['Training/Predict'] == 'training']

            # If it is first iteration
            if i == 1:
              # The feature is numeric, then impute it with the median
              if feature_to_be_imputed in columns_needed_to_be_encoded:
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
        imputed_feature = pd.concat([training[feature_to_be_imputed], predict[feature_to_be_imputed]], axis=0).sort_index()

        if feature_to_be_imputed in columns_needed_to_be_encoded:
          imputed_feature.replace(impute_map, inplace=True)

        return imputed_feature

    def impute(self, x, classifier, regressor, max_iter=5):
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
          imputed_feature = self.single_impute(x, feature, classifier, max_iter)

        else:
          imputed_feature = self.single_impute(x, feature, regressor, max_iter) 

        # Drop original features and Insert imputed features using the correct indexes
        insert_index = list(x.columns).index(feature)
        imputed_x.drop(feature, axis=1, inplace=True)
        imputed_x.insert(insert_index, feature, imputed_feature)  

      return imputed_x
