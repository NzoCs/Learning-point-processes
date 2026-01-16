import torch
from geomloss import SamplesLoss
from typing import Literal

from new_ltpp.shared_types import Batch, SimulationResult
from .kernel_abc import KernelABC


LossEnum = Literal["energy", "sinkhorn", "hausdorff", "gaussian", "laplacian"]


class MKernel(KernelABC):
    def __init__(
        self, gamma: int, blur: float, p: Literal[1, 2], loss: LossEnum = "gaussian"
    ):
        """Initialize the MMD kernel. Gamma is a parameter for the point process kernel,
        blur is a parameter for the underlying geomloss kernel.
        Args:
            gamma: The scale in the radial basis function of the point process kernel.
            blur: The finest level of detail that
                    should be handled by the loss function - in
                    order to prevent overfitting on the samples' locations.
            p: If **loss** is ``"sinkhorn"`` or ``"hausdorff"``,
                specifies the ground cost function between points.
                The supported values are: 1 or 2.
            loss: The loss function to use. Default is "gaussian".
        """
        self.gamma = gamma
        self.mmd_loss = SamplesLoss(loss=loss, p=p, blur=blur)

    def __call__(
        self, phi: Batch | SimulationResult, psi: Batch | SimulationResult
    ) -> torch.Tensor:
        # GeomLoss calcule la distance MMD^2
        dist_sq = self.mmd_loss(phi, psi)

        # On applique l'exponentielle pour avoir le noyau
        return torch.exp(-self.gamma * dist_sq)
