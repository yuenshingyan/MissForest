# MissForest
Arguably the best missing values imputation method.

MissForest aims to provide the most convenient way for the data science community to perform nonparametric imputation on missing values by using machine learning models.

- **Examples:** https://github.com/HindyDS/MissFores/tree/main/examples
- **Email:** hindy888@hotmail.com
- **Source code:** https://github.com/HindyDS/MissForest/tree/main/MissForest
- **Bug reports:** https://github.com/HindyDS/MissForest/issues
#
If you have any ideas for this packge please don't hesitate to bring forward!

# Flexible
You can implement other machine learning models besides RandomForest into MissForest

# Quick Start
    !pip install MissForest
    
    from missforest.miss_forest import MissForest

    mf = MissForest()
    mf.fit_transform(x)

    # return imputed dataframe
