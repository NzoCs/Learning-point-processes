import torch
from .protocol import ISpaceKernel


class EmbeddingKernel(ISpaceKernel):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 8,
    ):
        self.emb = torch.nn.Embedding(num_classes, embedding_dim)

    @torch.compile
    def Gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        e_X = self.emb(X)  # (B1, L, D)
        e_Y = self.emb(Y)  # (B2, K, D)
        dist = torch.cdist(e_X.view(-1, e_X.shape[-1]), e_Y.view(-1, e_Y.shape[-1]))
        dist_sq = dist.pow(2).view(X.shape[0], Y.shape[0], X.shape[1], Y.shape[1])
        return torch.exp(-dist_sq / 2)

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        e_X = self.emb(X)  # (B, L, D)
        e_Y = self.emb(Y)  # (B, K, D)
        dist = torch.cdist(e_X, e_Y)  # (B, L, K)
        dist_sq = dist.pow(2)
        return torch.exp(-dist_sq / 2)
