import torch
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.models.base_model import NeuralModel
from new_ltpp.data.preprocess.data_loader import TypedDataLoader

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

    def test(self, model: NeuralModel, data_loader: TypedDataLoader) -> float:
        """Compute the MMD over the entire dataset using the provided model. Supposed to be used to test an
        already trained model.
        args:
            model: The neural point process model to evaluate.
            data_loader: DataLoader providing batches of real sequences.
        returns:
            The average MMD value over the dataset.
        """

        mmd = torch.tensor(0.0, device=next(model.parameters()).device)
        for batch in data_loader:
            phi = model.simulate(batch=batch)
            mmd += self(phi, batch)
        return mmd.item() / len(data_loader)

    def evaluate(self, model: NeuralModel, batch: Batch) -> float:
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

        return mmd.item()
