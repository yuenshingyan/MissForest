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
    
    def impute(self, df, impute, model, max_iter=5, verbose=0):
        df3 = df.copy()
        pd.options.mode.chained_assignment = None  # default='warn'
                  
        encode_col = [f for f in df.columns if isinstance(df[f].sample(1).values[0], str)]
        for c in encode_col:  
            df3[c].replace(df[c].dropna().unique(), range(df[c].dropna().nunique()), inplace=True) # label encoding
            if c == impute:
                impute_map = ({k:v for (k,v) in zip(df[c].dropna().unique(), range(df[c].dropna().nunique()))})
                impute_map = {v:k for (k,v) in impute_map.items()}
       
        non_null = list(set([feat for feat, val in zip(df.isnull().sum().index, df.isnull().sum()) if val == 0] + [impute]))
            
        track = {}
        for n_iter in range(max_iter):
            df2=df3[non_null].copy()
            df2['Training/Predict'] = np.where(df2[impute].isnull(), 'predict', 'training')
            predict_set = df2[df2['Training/Predict'] == 'predict']
            training_set = df2[df2['Training/Predict'] == 'training']

            if n_iter == 1:
                predict_set[impute].fillna(df2[impute].median(), inplace=True) if impute in encode_col else predict_set[impute].fillna(df2[impute].mode().values[0], inplace=True)
                   
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
