from tests.integration_tests import IntegrationTests
from tests.unit_tests.missforest_class_methods import MissForestClassMethods
from tests.unit_tests.is_numerical_matrix import IsNumericalMatrix
from tests.unit_tests.validate_2d import Validate2D
from tests.unit_tests.validate_cat_var_consistency import (
    ValidateCategoricalVariableConsistency
)
from tests.unit_tests.validate_categorical import ValidateCategorical
from tests.unit_tests.validate_clf import ValidateClassifier
from tests.unit_tests.validate_consistent_dimensions import (
    ValidateConsistentDimensions
)
from tests.unit_tests.validate_early_stopping import ValidateEarlyStopping
from tests.unit_tests.validate_empty_feature import ValidateEmptyFeature
from tests.unit_tests.validate_feature_dtype_consistency import (
    FeatureDataType
)
from tests.unit_tests.validate_imputable import ValidateImputable
from tests.unit_tests.validate_infinite import ValidateInfinite
from tests.unit_tests.validate_initial_guess import ValidateInitialGuess
from tests.unit_tests.validate_max_iter import ValidateMaxIter
from tests.unit_tests.validate_rgr import ValidateRegressor
from tests.unit_tests.validate_verbose import ValidateVerbose
import unittest


if __name__ == "__main__":
    unittest.main()
