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
from .linear_kernel import LinearKernel


class TimeKernelType(Enum):
    RBF = "rbf"
    IMQ = "imq"
    MATERN_3_2 = "matern_3_2"
    MATERN_5_2 = "matern_5_2"
    LAPLACIAN = "laplacian"
    RATIONAL_QUADRATIC = "rq"
    BSPLINE = "bspline"
    POISSON = "poisson"
    LINEAR = "linear"


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
    "LinearKernel",
]
