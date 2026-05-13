import torch
from .protocol import ISpaceKernel


class EmbeddingKernel(ISpaceKernel):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 8,
    ):
        self.emb = torch.nn.Embedding(num_classes + 1, embedding_dim)

    def _emb_on_device(self, x: torch.Tensor) -> torch.Tensor:
        """Move embedding to x's device if needed, clamp indices, then embed."""
        if next(self.emb.parameters()).device != x.device:
            self.emb = self.emb.to(x.device)
        num_emb = self.emb.num_embeddings
        if (x < 0).any() or (x >= num_emb).any():
            import warnings
            warnings.warn(
                f"EmbeddingKernel received out-of-range indices "
                f"(min={x.min().item()}, max={x.max().item()}, num_classes={num_emb}). "
                f"Clamping to [0, {num_emb - 1}]. Check your type_seqs / num_classes config.",
                stacklevel=2,
            )
            x = x.clamp(0, num_emb - 1)
        return self.emb(x)

    @torch.compile
    def Gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        e_X = self._emb_on_device(X)  # (B1, L, D)
        e_Y = self._emb_on_device(Y)  # (B2, K, D)
        dist = torch.cdist(e_X.view(-1, e_X.shape[-1]), e_Y.view(-1, e_Y.shape[-1]))
        dist_sq = dist.pow(2).view(X.shape[0], Y.shape[0], X.shape[1], Y.shape[1])
        return torch.exp(-dist_sq / 2)

    @torch.compile
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        e_X = self._emb_on_device(X)  # (B, L, D)
        e_Y = self._emb_on_device(Y)  # (B, K, D)
        dist = torch.cdist(e_X, e_Y)  # (B, L, K)
        dist_sq = dist.pow(2)
        return torch.exp(-dist_sq / 2)
