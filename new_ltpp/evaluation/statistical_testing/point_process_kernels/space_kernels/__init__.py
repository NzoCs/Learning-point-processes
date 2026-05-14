from enum import Enum

from .protocol import ISpaceKernel

from .rbf import RBFKernel
from .embedding import EmbeddingKernel
from .linear_kernel import LinearKernel


class TimeKernelType(Enum):
    RBF = "rbf"
    LINEAR = "linear"


__all__ = [
    "ISpaceKernel",
    "RBFKernel",
    "EmbeddingKernel",
    "TimeKernelType",
    "LinearKernel",
]
