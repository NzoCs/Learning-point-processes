from new_ltpp.statistical_testing.statistical_testing_interface import Kernel
from new_ltpp.models.neural_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader
from abc import ABC, abstractmethod


class TestABC(ABC):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def test(self, model: NeuralModel, dataset: TypedDataLoader) -> bool:
        """Perform a goodness of fit test. The null hypothesis is that the model is a good fit for the data.
        If we fail to reject the null hypothesis, we can say that the model is a good fit for the data.
        Args:
            model: The model to test.
            dataset: The dataset to test on.
        Returns:
            False if the null hypothesis is rejected, True otherwise.
        """
        pass

    @abstractmethod
    def p_value(self, model: NeuralModel, dataset: TypedDataLoader) -> float:
        pass
