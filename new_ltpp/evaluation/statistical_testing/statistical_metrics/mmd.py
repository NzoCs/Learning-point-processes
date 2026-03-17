import torch
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.models.base_model import NeuralModel

from .base_stat_metric import StatMetric


class MMD(StatMetric):
    @torch.compile
    def __call__(
        self, phi: Batch | SimulationResult, psi: Batch | SimulationResult
    ) -> torch.Tensor:
        """Compute the Maximum Mean Discrepancy (MMD) between two batches of sequences.
        args:
            phi: Batch of sequences of shape (B1, L)
            psi: Batch of sequences of shape (B2, K)
        returns:
            The MMD value as a torch.Tensor.
        """

        B1, L = phi.time_seqs.shape
        B2, K = psi.time_seqs.shape

        mmd_value = self.compute_mmd(phi, psi)
        return mmd_value

    def evaluate(self, model: NeuralModel, batch: Batch) -> torch.Tensor:
        """Compute the MMD between the model's simulated sequences and the real sequences in the batch.
        args:
            model: The neural point process model to evaluate.
            batch: Batch of real sequences.
        returns:
            The MMD value as a float aggregated over the batch.
        """

        phi = model.simulate(batch=batch)
        psi = batch
        mmd = self(phi, psi)

        return mmd
