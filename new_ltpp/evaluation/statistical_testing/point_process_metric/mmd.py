import torch

from new_ltpp.shared_types import Batch, SimulationResult

from .base_stat_metric import IStatMetric, StatMetric


class MMD(StatMetric):
    def __call__(
        self, X: Batch | SimulationResult, Y: Batch | SimulationResult
    ) -> torch.Tensor:
        """Compute off-diagonal unbiased MMD squared; finite estimates may be negative.
        args:
            X: Batch of sequences of shape (B1, L)
            Y: Batch of sequences of shape (B2, K)
        returns:
            The MMD value as a torch.Tensor.
        """
        return self.compute_mmd(X, Y)


# type checks
if __name__ == "__main__":
    from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
        LinearKernel,
    )

    mmd: IStatMetric = MMD(
        SIGKernel(
            static_kernel=LinearKernel(),
            embedding_type="counting_grid",
            num_discretization_points=100,
            dyadic_order=3,
            num_event_types=10,
        )
    )
