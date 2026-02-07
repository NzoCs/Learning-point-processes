from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from .test_protocol import TestProtocol


class SteinTest(TestProtocol):
    def __init__(self, kernel):
        self.kernel = kernel

    def p_value(self, model: NeuralModel, dataset: TypedDataLoader):
        # TODO: implement the p-value computation for Stein test
