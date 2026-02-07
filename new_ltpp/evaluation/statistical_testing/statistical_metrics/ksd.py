import torch
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.models.base_model import NeuralModel
from .protocols import StatMetricsProtocol


class KSD(StatMetricsProtocol):
    def __call__(self, model: NeuralModel, batch: Batch) -> float:
        pass  # TODO: Implement KSD metric calculation

    @staticmethod
    def batched_insert_samples(
        seqs: torch.Tensor, idx: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Insert samples into sequences at specified indices.
        Args:
            seqs: (B, L)
            idx: (B, N) indices where to insert new values
            values: (B, N) values to insert
        Returns:
            (B, L + 1, N) sequences with inserted values
        """
        B, L = seqs.shape
        N = idx.shape[1]
        ar = torch.arange(L + 1, device=seqs.device)  # positions 0..L

        # mask[b, j] = True si j > idx[b]
        mask = ar.unsqueeze(0).unsqueeze(-1) > idx.unsqueeze(1)  # (B, L+1, N)

        # Nouveau batch vide
        out = torch.empty((B, L + 1, N), device=seqs.device, dtype=seqs.dtype)

        # 1. mettre la nouvelle valeur
        out[torch.arange(B), idx, :] = values

        # 2. copier l'ancienne séquence en décalant après idx
        # positions > idx, ici on veut seq[b, j-1] → out[b, j], car les seqs sont décalées d'1 vers la droite
        out[:, 1:, :] = torch.where(
            mask[:, 1:, :],  # positions > idx
            seqs,  # on prend seq[b, j-1]
            out[:, 1:, :],  # sinon on garde le déjà rempli (new value)
        )

        # 3. copier la partie avant idx
        # mask[:, :-1] = positions < idx
        # positions < idx, on veut seq[b, j] → out[b, j] car avant idx pas de décalage
        before_mask = ar.unsqueeze(0).unsqueeze(-1) < idx.unsqueeze(1)
        out[before_mask] = seqs[before_mask]

        return out

    @staticmethod
    def batched_delete(seqs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        seqs: (B, L)
        idx: (B,)
        return: (B, L-1) avec value en idx[b] supprimé
        """

        B, L = seqs.shape
        ar = torch.arange(L, device=seqs.device)  # positions 0..L-1

        # mask[b, j] = True si j > idx[b]
        mask = ar.unsqueeze(0) > idx.unsqueeze(1)  # (B, L)

        # Nouveau batch vide
        out = torch.empty((B, L - 1), device=seqs.device, dtype=seqs.dtype)

        # 1. copier l'ancienne séquence en décalant après idx
        # positions >= idx, ici on veut seq[b, j+1] → out[b, j], car les seqs sont décalées d'1 vers la gauche
        out = torch.where(
            mask,  # positions >= idx
            seqs[:, 1:],  # on prend seq[b, j+1]
            out,  # sinon on garde le déjà rempli
        )

        # 2. copier la partie avant idx
        # mask[:, :-1] = positions < idx
        # positions < idx, on veut seq[b, j] → out[b, j] car avant idx pas de décalage
        before_mask = ar.unsqueeze(0) < idx.unsqueeze(1)
        out[before_mask] = seqs[before_mask]

        return out

    def _first_integrand(
        self,
        model: NeuralModel,
        phi_time_seqs: torch.Tensor,
        phi_type_seqs: torch.Tensor,
        psi_time_seqs: torch.Tensor,
        psi_type_seqs: torch.Tensor,
    ) -> float:
        """This is the function f(u,v) = [k(phi + delta_u, psi + delta_v) - k(phi,psi + delta_v)
        - k(phi + delta_u, psi) + k(phi, psi)]*roh(u|phi)roh(v|psi).
        args:
            model: NeuralModel
            phi_time_seqs: (B, L, N)
            phi_type_seqs: (B, L, N)
            psi_time_seqs: (B, K, N)
            psi_type_seqs: (B, K, N)
        returns:
            The approximation of the first integral over u and v.
        """

    def _first_integral(
        self,
        model: NeuralModel,
        phi: Batch | SimulationResult,
        psi: Batch | SimulationResult,
    ) -> torch.Tensor:
        pass  # TODO: Implement the first integral calculation
