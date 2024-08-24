# MissForest

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13368883.svg)](https://doi.org/10.5281/zenodo.13368883)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MissForest?link=https%3A%2F%2Fpypi.org%2Fproject%2FMissForest%2F)


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

Categorical variables in argument `categoricals` will be label encoded for
estimators to work properly. 

# Example

To install MissForest using pip.

```console
pip install MissForest
```

Imputing a dataset:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from missforest import MissForest

# Load toy dataset.
df = pd.read_csv("insurance.csv")

# Label encoding.
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["region"] = df["region"].map({
    "southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3})

# Create missing values.
for c in df.columns:
    n = int(len(df) * 0.1)
    rand_idx = np.random.choice(df.index, n)
    df.loc[rand_idx, c] = np.nan

# Split dataset into train and test sets.
train, test = train_test_split(df, test_size=.3, shuffle=True,
                               random_state=42)

# Default estimators are lgbm classifier and regressor
mf = MissForest()
mf.fit(
    x=train,
    categorical=["sex", "smoker", "region"]
)
train_imputed = mf.transform(x=train)
test_imputed = mf.transform(x=test)
```

Or using the `fit_transform` method
```python
mf = MissForest()
train_imputed = mf.fit_transform(
    X=train,
    categorical=["sex", "smoker", "region"]
)
test_imputed = mf.transform(X=test)
print(test_imputed)
```

# Imputing with other estimators

```python
from missforest import MissForest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

df = pd.read_csv("insurance.csv")

for c in df.columns:
    random_index = np.random.choice(df.index, size=100)
    df.loc[random_index, c] = np.nan

clf = RandomForestClassifier(n_jobs=-1)
rgr = RandomForestRegressor(n_jobs=-1)

mf = MissForest(clf, rgr)
df_imputed = mf.fit_transform(df)
```



# Benchmark

Mean Absolute Percentage Error
|         | missForest | mean/mode | Difference |
|:-------:|:----------:|:---------:|:----------:|
| charges | 2.65%      | 9.72%     |  -7.07%    |
| age     | 1.16%      | 2.77%     |  -1.61%    |
| bmi     | 1.18%      | 1.25%     |  -0.07%    |
| sex     | 21.21      | 31.82     |  -10.61    |
| smoker  |  4.24      |  9.90     |   -5.66    |
| region  | 46.67      | 38.96     |   +7.71    |
