import torch
import torch.nn as nn
from .protocol import ISpaceKernel

class RBFKernel(ISpaceKernel):
    """RBF kernel k: R^d x R^d -> R"""
    def __init__(
        self,
        sigma: float = 1.0,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.scaling = scaling

    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]

        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))
        return self.scaling * torch.exp(-dist/self.sigma)

    def Gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]

        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))
        return self.scaling * torch.exp(-dist/self.sigma)
