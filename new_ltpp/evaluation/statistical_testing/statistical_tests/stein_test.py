from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from .base_test import Test


class SteinTest(Test):
    def __init__(self, kernel):
        self.kernel = kernel

    def p_value(self, model: NeuralModel, dataset: TypedDataLoader):
        # TODO: implement the p-value computation for Stein test
        pass
