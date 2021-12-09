#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

class MissForestExtra:
    def __init__(self):     
        self.__author__ = 'Yuen Shing Yan Hindy'
        self.__license__= 'MIT License'
        self.__contact__ = 'https://github.com/HindyDS/MissForest'
    
    def single_impute(self, x, impute, model, max_iter=5, verbose=0):
        x3 = x.copy()
        pd.options.mode.chained_assignment = None  # default='warn'
                  
        encode_col = [f for f in x.columns if isinstance(x[f].sample(1).values[0], str)]
        for c in encode_col:  
            x3[c].replace(x[c].dropna().unique(), range(x[c].dropna().nunique()), inplace=True) # label encoding
            if c == impute:
                impute_map = ({k:v for (k,v) in zip(x[c].dropna().unique(), range(x[c].dropna().nunique()))})
                impute_map = {v:k for (k,v) in impute_map.items()}
       
        non_null = list(set([feat for feat, val in zip(x.isnull().sum().index, x.isnull().sum()) if val == 0] + [impute]))
            
        track = {}
        for n_iter in range(max_iter):
            x2=x3[non_null].copy()
            x2['Training/Predict'] = np.where(x2[impute].isnull(), 'predict', 'training')
            predict_set = x2[x2['Training/Predict'] == 'predict']
            training_set = x2[x2['Training/Predict'] == 'training']

            if n_iter == 1:
                predict_set[impute].fillna(x2[impute].median(), inplace=True) if impute in encode_col else predict_set[impute].fillna(x2[impute].mode().values[0], inplace=True)
                   
            if n_iter > 1:
                for idx, pred in zip(predict_set.index, res):
                    predict_set.loc[idx, impute] = pred
           
            model.fit(training_set[non_null].drop(impute, axis=1), training_set[impute])
            if predict_set[non_null].drop(impute, axis=1).shape[0] != 0:
                res = model.predict(predict_set[non_null].drop(impute, axis=1))
            else:
                raise Exception(f'Feature "{impute}" does not contain any missing values.')
                
            track[f'iter{n_iter}'] = res
            
            if n_iter > 1:       
                if (track[f'iter{n_iter}'] == track[f'iter{n_iter - 1}']).all():
                    if verbose == 1:
                        print('Stopping Criteria Reached (no change in imputed values)')
                    break
            if verbose == 1:       
                if max_iter <= 1:
                    print('Stopping Criteria Reached (max iter reached)')
                    break
                   
            n_iter += 1
           
        pd.options.mode.chained_assignment = 'warn'  
       
        imputed_feature = pd.concat([training_set[impute], predict_set[impute]], axis=0).sort_index()
        return imputed_feature.replace(impute_map) if impute in encode_col else imputed_feature

    def impute(self, x, classifier, regressor, max_iter=5, verbose=0):
      x = x.copy()
      x2 = x.copy()
      columns_that_require_imputation_in_ascending_order = x.isnull().sum()[x.isnull().sum() != 0].sort_values().index

      is_string = {}
      for col in columns_that_require_imputation_in_ascending_order:
        is_string[col] = isinstance(x[~x[col].isnull()].loc[:, col].sample(1).values[0], str)
        
      for col in columns_that_require_imputation_in_ascending_order:
        if is_string[col]:
          imputed = self.single_impute(x, col, classifier, max_iter, verbose)

        else:
          imputed = self.single_impute(x, col, regressor, max_iter, verbose) 

        insert_index = list(x.columns).index(col)
        x2.drop(col, axis=1, inplace=True)
        x2.insert(insert_index, col, imputed)  

      return x2
