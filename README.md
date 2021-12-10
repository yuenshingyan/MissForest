# MissForest
Arguably the best missing values imputation method.

MissForest aims to provide the most convenient way for the data science community to perform nonparametric imputation on missing values by using machine learning models.

- **Examples:** https://github.com/HindyDS/MissFores/tree/main/examples
- **Email:** hindy888@hotmail.com
- **Source code:** https://github.com/HindyDS/MissForest/tree/main/MissForest
- **Bug reports:** https://github.com/HindyDS/MissForest/issues
- 
# Convenient
It only requires 3 arguments to run:

- x: dataset that being imputed
- feature_to_be_imputed (str): feature that being imputed
- estimator: machine learning model

Optional arguments:
- max_iter (int): maximum number of iterations

If you have any ideas for this packge please don't hesitate to bring forward!

# Flexible
You can implement other machine learning models besides RandomForest into MissForest

# Quick Start
    !pip install MissFores
    
    from MissForest import MissForest

    mfe = MissForest()

    mfe.single_impute(x, feature_to_be_imputed, estimator)

    # return the imputed pandas series

    mfe.impute(x, classifier, regressor)

     # return imputed dataframe
