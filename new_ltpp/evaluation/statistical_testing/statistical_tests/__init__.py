from .base_test import ITest
from .factory import create_statistical_test
from .mmd_test import MMDTwoSampleTest

__all__ = ["MMDTwoSampleTest", "ITest", "create_statistical_test"]
