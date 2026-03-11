import torch
from .protocol import ISpaceKernel

class LinearKernel(ISpaceKernel):
    """Linear kernel k: R^d x R^d -> R"""

    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling
        
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        return self.scaling * torch.bmm(X, Y.permute(0,2,1))

    def Gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        return self.scaling * torch.einsum('ipk,jqk->ijpq', X, Y)
