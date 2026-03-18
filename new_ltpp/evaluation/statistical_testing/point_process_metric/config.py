class PointProcessMetricConfig:
    """Configuration for kernel used in statistical tests.

    Args:
        kernel_type: Type of kernel to use. One of "bspline", "poisson", "embedding", "linear".
        embedding_dim: Dimension of the embedding space (only for "embedding" kernel).
        sigma: Bandwidth parameter for RBF kernels (only for "embedding" kernel).
        scaling: Scaling factor for the kernel (only for "embedding" kernel).
    """

    def __init__(
        self,
        kernel_type: str,
        embedding_dim: int = 8,
        sigma: float = 1.0,
        scaling: float = 1.0,
    ):
        self.kernel_type = kernel_type
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        self.scaling = scaling
