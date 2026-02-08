from enum import Enum

from .protocol import ISpaceKernel

from .rbf import RBFTimeKernel
from .imq import IMQTimeKernel
from .matern import MaternTimeKernel
from .laplacian import LaplacianTimeKernel
from .rational_quadratic import RationalQuadraticTimeKernel
from .bspline import BSplineTimeKernel
from .poisson import PoissonTimeKernel
from .embedding import EmbeddingKernel


class TimeKernelType(Enum):
    RBF = "rbf"
    IMQ = "imq"
    MATERN_3_2 = "matern_3_2"
    MATERN_5_2 = "matern_5_2"
    LAPLACIAN = "laplacian"
    RATIONAL_QUADRATIC = "rq"
    BSPLINE = "bspline"
    POISSON = "poisson"


def create_time_kernel(
    kernel_type: TimeKernelType | str, sigma: float = 1.0, **kwargs
) -> ISpaceKernel:
    if isinstance(kernel_type, str):
        kernel_type = TimeKernelType(kernel_type)

    if kernel_type == TimeKernelType.RBF:
        return RBFTimeKernel(sigma=sigma)
    if kernel_type == TimeKernelType.IMQ:
        return IMQTimeKernel(c=kwargs.get("c", 1.0), beta=kwargs.get("beta", 0.5))
    if kernel_type == TimeKernelType.MATERN_3_2:
        return MaternTimeKernel(sigma=sigma, nu=1.5)
    if kernel_type == TimeKernelType.MATERN_5_2:
        return MaternTimeKernel(sigma=sigma, nu=2.5)
    if kernel_type == TimeKernelType.LAPLACIAN:
        return LaplacianTimeKernel(sigma=sigma)
    if kernel_type == TimeKernelType.RATIONAL_QUADRATIC:
        return RationalQuadraticTimeKernel(sigma=sigma, alpha=kwargs.get("alpha", 1.0))
    if kernel_type == TimeKernelType.BSPLINE:
        return BSplineTimeKernel(sigma=sigma, order=kwargs.get("order", 3))
    if kernel_type == TimeKernelType.POISSON:
        return PoissonTimeKernel(r=kwargs.get("r", 0.5), d=kwargs.get("d", 1))

    raise ValueError(f"Unknown kernel type: {kernel_type}")


__all__ = [
    "ISpaceKernel",
    "RBFTimeKernel",
    "IMQTimeKernel",
    "MaternTimeKernel",
    "LaplacianTimeKernel",
    "RationalQuadraticTimeKernel",
    "BSplineTimeKernel",
    "PoissonTimeKernel",
    "EmbeddingKernel",
    "TimeKernelType",
    "create_time_kernel",
]
