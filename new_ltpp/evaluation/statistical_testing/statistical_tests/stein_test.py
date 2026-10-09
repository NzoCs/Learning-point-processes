from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from new_ltpp.models.base.base_model import NeuralModel


class SteinTest:
    def __init__(self, kernel):
        self.kernel = kernel

    def p_value(self, model: NeuralModel, dataset: TypedDataLoader):
        # TODO: implement the p-value computation for Stein test
        pass
