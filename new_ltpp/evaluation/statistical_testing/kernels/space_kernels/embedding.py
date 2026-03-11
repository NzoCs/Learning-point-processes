import torch
from .protocol import ISpaceKernel


class EmbeddingKernel(ISpaceKernel):
    def __init__(self, num_classes: int, embedding_dim: int, sigma: float = 1.0, scaling: float = 1.0):
        self.emb = torch.nn.Embedding(num_classes, embedding_dim)
        self.sigma = sigma
        self.scaling = scaling

    @torch.compile
    def Gram_matrix(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        e_X = self.emb(X)  # (B1, L, D)
        e_Y = self.emb(Y)  # (B2, K, D)
        dist = torch.cdist(
            e_X.view(-1, e_X.shape[-1]), e_Y.view(-1, e_Y.shape[-1])
        )
        dist_sq = dist.pow(2).view(
            X.shape[0], Y.shape[0], X.shape[1], Y.shape[1]
        )
        return self.scaling * torch.exp(-dist_sq / (2 * self.sigma**2))

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        e_X = self.emb(X)  # (B, L, D)
        e_Y = self.emb(Y)  # (B, K, D)
        B, L, D = e_X.shape
        # Compute pairwise distances for each batch
        dist_sq_list = []
        for b in range(B):
            dist = torch.cdist(e_X[b : b + 1], e_Y[b : b + 1])  # (1, L, K)
            dist_sq_list.append(dist.pow(2).squeeze(0))  # (L, K)
        dist_sq = torch.stack(dist_sq_list, dim=0)  # (B, L, K)
        return self.scaling * torch.exp(-dist_sq / (2 * self.sigma**2))
