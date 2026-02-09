import torch
from .protocol import ISpaceKernel


class EmbeddingKernel(ISpaceKernel):
    def __init__(self, num_classes: int, embedding_dim: int, sigma: float = 1.0):
        self.emb = torch.nn.Embedding(num_classes, embedding_dim)
        self.sigma = sigma

    @torch.compile
    def cross_batch_kernel_matrix(
        self, phi: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        e_phi = self.emb(phi)  # (B1, L, D)
        e_psi = self.emb(psi)  # (B2, K, D)
        dist = torch.cdist(
            e_phi.view(-1, e_phi.shape[-1]), e_psi.view(-1, e_psi.shape[-1])
        )
        dist_sq = dist.pow(2).view(
            phi.shape[0], psi.shape[0], phi.shape[1], psi.shape[1]
        )
        return torch.exp(-dist_sq / (2 * self.sigma**2))

    @torch.compile
    def intra_batch_kernel_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        e_phi = self.emb(phi)  # (B, L, D)
        B, L, D = e_phi.shape
        # Compute pairwise distances for each batch
        dist_sq_list = []
        for b in range(B):
            dist = torch.cdist(e_phi[b : b + 1], e_phi[b : b + 1])  # (1, L, L)
            dist_sq_list.append(dist.pow(2).squeeze(0))  # (L, L)
        dist_sq = torch.stack(dist_sq_list, dim=0)  # (B, L, L)
        return torch.exp(-dist_sq / (2 * self.sigma**2))
