# MissForest
This project is a Python implementation of the MissForest algorithm, a powerful 
tool designed to handle missing values in tabular datasets. The primary goal of 
this project is to provide users with a more accurate method of imputing 
missing data.

While MissForest may take more time to process datasets compared to simpler 
imputation methods, it typically yields more accurate results.

Please note that the efficiency of MissForest is a trade-off for its accuracy. 
It is designed for those who prioritize data accuracy over processing speed. 
This makes it an excellent choice for projects where the quality of data is 
paramount.

# How MissForest Handles Categorical Variables ?
Categorical variables in argument 'categoricals' will be label encoded for
estimators to work properly. 

# Example
To install MissForest using pip.

    pip install MissForest

Imputing a dataset:

    from missforest.missforest import MissForest
    import pandas as pd
    import numpy as np
    
    
    if __name__ == "__main__":
        df = pd.read_csv("insurance.csv")

        # default estimators are lgbm classifier and regressor
        mf = MissForest()
        mf.fit(
            X=train,
            categorical=["sex", "smoker", "region"]
        )
        train_imputed = mf.transform(X=train)
        test_imputed = mf.transform(X=test)
        print(test_imputed)

        # or using the 'fit_transform' method
        mf = MissForest()
        train_imputed = mf.fit_transform(
            X=train,
            categorical=["sex", "smoker", "region"]
        )
        test_imputed = mf.transform(X=test)
        print(test_imputed)

# Imputing with other estimators

    from missforest.missforest import MissForest
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    
    if __name__ == "__main__":
        df = pd.read_csv("insurance.csv")
        df_or = df.copy()
        for c in df.columns:
            random_index = np.random.choice(df.index, size=100)
            df.loc[random_index, c] = np.nan

    clf = RandomForestClassifier(n_jobs=-1)
    rgr = RandomForestRegressor(n_jobs=-1)

    mf = MissForest(clf, rgr)
    df_imputed = mf.fit_transform(df)



# Benchmark

                Mean Absolute Percentage Error
               missForest | mean/mode | Difference
     charges        2.65%       9.72%       -7.07%
         age        1.16%       2.77%       -1.61%
         bmi        1.18%       1.25%       -0.07%
         sex        21.21       31.82       -10.61
      smoker         4.24        9.90        -5.66
      region        46.67       38.96        +7.71
