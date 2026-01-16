from new_ltpp.evaluation.statistical_testing.testing_protocols import Kernel
from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from .test_abc import TestABC


class SteinTest(TestABC):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def test(self, model: NeuralModel, dataset: TypedDataLoader):
        pass

    def p_value(self, model: NeuralModel, dataset: TypedDataLoader):
        pass
