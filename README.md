# MissForestExtra
Arguably the best missing values imputation method.

MissForestExtra aims to provide the most convenient way for the data science community to perform nonparametric imputation on missing values by using machine learning models.

- **Examples:** https://github.com/HindyDS/MissForestExtra/tree/main/examples
- **Email:** hindy888@hotmail.com
- **Source code:** https://github.com/HindyDS/MissForestExtra/tree/main/MissForestExtra 
- **Bug reports:** https://github.com/HindyDS/MissForestExtra/issues
- 
# Convenient
It only requires 3 arguments to run:

For single_impute:
- df: dataset that being imputed
- impute (str): feature that being imputed
- model: machine learning model


For impute:
- df: dataset that being imputed
- classifier: sklearn classifier
- regressor: sklearn regressor

Optional arguments:
- max_iter (int): maximum number of iterations
- verbose (int): Level of verbosity of MFE

If you have any ideas for this packge please don't hesitate to bring forward!

# Flexible
You can implement other machine learning models besides RandomForest into MissForestExtra

# Quick Start
    !pip install MissForestExtra
    
    from MissForestExtra import MissForestExtra

    mfe = MissForestExtra()

    mfe.singe_impute(df, impute, model)

    # return the imputed pandas series
    
    mfe.impute(df, classifier, regressor)
    
    # return imputed dataframe
