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

        k_xx = self.compute_kernel_matrix(phi, phi)  # (B1, B1)
        k_yy = self.compute_kernel_matrix(psi, psi)  # (B2, B2)
        k_xy = self.compute_kernel_matrix(phi, psi)  # (B1, B2)

        XX_reg = torch.max(
            torch.tensor(B1 * (B1 - 1), device=phi.time_seqs.device),
            torch.tensor(1.0, device=phi.time_seqs.device),
        )
        YY_reg = torch.max(
            torch.tensor(B2 * (B2 - 1), device=psi.time_seqs.device),
            torch.tensor(1.0, device=psi.time_seqs.device),
        )

        mmd_value = (
            (k_xx.sum() - k_xx.diagonal().sum()) / XX_reg
            + (k_yy.sum() - k_yy.diagonal().sum()) / YY_reg
            - 2 * k_xy.sum() / (B1 * B2)
        )
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
