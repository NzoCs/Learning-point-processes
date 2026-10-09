"""Explicit signature backends; the historical MMD estimator is preserved."""

import math
from importlib.metadata import PackageNotFoundError, version

import pysiglib
import pysiglib.torch_api as pysiglib_torch
import torch

from .space_kernels import LinearKernel, RBFKernel


class _ScaledStaticKernel(pysiglib.StaticKernel):
    """Scale the static kernel, not the final signature Gram.

    Each call owns its inner context. Keeping the scale outside the inner
    gradients also preserves RBFKernel's cached derivative for the second path.
    """

    def __init__(self, kernel, scaling):
        self.kernel = kernel
        self.scaling = scaling

    def __call__(self, ctx, x, y):
        inner = pysiglib.Context()
        ctx.save_for_backward(inner)
        return self.scaling * self.kernel(inner, x, y)

    def grad_x(self, ctx, derivs):
        return self.scaling * self.kernel.grad_x(ctx.saved_tensors[0], derivs)

    def grad_y(self, ctx, derivs):
        return self.scaling * self.kernel.grad_y(ctx.saved_tensors[0], derivs)


def _translate_static_kernel(kernel):
    if not isinstance(kernel, (LinearKernel, RBFKernel)):
        raise TypeError(
            "pySigLib adapter supports only the project's LinearKernel and RBFKernel."
        )
    scaling = float(kernel.scaling)
    if not math.isfinite(scaling) or scaling < 0:
        raise ValueError(
            "Signature static-kernel scaling must be finite and non-negative."
        )
    if isinstance(kernel, RBFKernel):
        sigma = float(kernel.sigma)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("RBF sigma must be finite and positive.")
        base = pysiglib.RBFKernel(sigma=sigma)
    else:
        base = pysiglib.LinearKernel()
    return base if scaling == 1.0 else _ScaledStaticKernel(base, scaling)


def unbiased_mmd_squared(kxx, kxy, kyy):
    """Same off-diagonal MMD² as sigkernel at commit 40a5831; may be negative."""
    n, m = kxy.shape
    if n < 2 or m < 2:
        raise ValueError(
            "Unbiased signature MMD requires at least two paths per sample."
        )
    return (
        (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        - 2 * kxy.mean()
    )


class PySigLibKernel:
    def __init__(self, static_kernel, dyadic_order, max_batch=64):
        if (
            not isinstance(dyadic_order, int)
            or isinstance(dyadic_order, bool)
            or dyadic_order < 0
        ):
            raise ValueError("dyadic_order must be a non-negative integer.")
        if (
            not isinstance(max_batch, int)
            or isinstance(max_batch, bool)
            or max_batch < 1
        ):
            raise ValueError("max_batch must be a positive integer.")
        self.static_kernel = _translate_static_kernel(static_kernel)
        self.dyadic_order = dyadic_order
        self.max_batch = max_batch
        self.kernel_parameters = {
            "type": type(static_kernel).__name__,
            "scaling": float(static_kernel.scaling),
        }
        if isinstance(static_kernel, RBFKernel):
            self.kernel_parameters["sigma"] = float(static_kernel.sigma)

    def compute_Gram(self, x, y):
        if x.device != y.device:
            raise ValueError("Signature paths must be on the same device.")
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("This signature profile supports only CPU and CUDA.")
        if x.ndim != 3 or y.ndim != 3 or x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "Expected paths shaped (batch, length, matching channels)."
            )
        if min(x.shape[0], y.shape[0]) < 1 or min(x.shape[1], y.shape[1]) < 2:
            raise ValueError(
                "Signature Gram requires non-empty batches and at least two path vertices."
            )
        if x.device.type == "cuda" and not pysiglib.BUILT_WITH_CUDA:
            raise RuntimeError(
                "pySigLib CUDA plugin unavailable. Install the locked Ruche extra."
            )
        # The project embedding already uses float64. Make this precision
        # explicit for direct callers too; .to() keeps the autograd connection.
        x = x.to(dtype=torch.float64).contiguous()
        y = y.to(dtype=torch.float64).contiguous()
        api = (
            pysiglib_torch
            if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad)
            else pysiglib
        )
        result = api.sig_kernel_gram(
            x,
            y,
            method="finite_difference",
            dyadic_order=self.dyadic_order,
            static_kernel=self.static_kernel,
            time_aug=False,
            lead_lag=False,
            normalize=False,
            n_jobs=1,
            max_batch=self.max_batch,
        )
        if result.device != x.device:
            raise RuntimeError("Signature backend returned a result on another device.")
        return result

    def compute_mmd(self, x, y):
        if min(x.shape[0], y.shape[0]) < 2:
            raise ValueError(
                "Unbiased signature MMD requires at least two paths per sample."
            )
        return unbiased_mmd_squared(
            self.compute_Gram(x, x), self.compute_Gram(x, y), self.compute_Gram(y, y)
        )

    def metadata(self):
        try:
            plugin_version = version("pysiglib-cuda")
        except PackageNotFoundError:
            plugin_version = None
        return {
            "backend": "pysiglib",
            "version": version("pysiglib"),
            "cuda_plugin_loaded": bool(pysiglib.BUILT_WITH_CUDA),
            "cuda_plugin_version": plugin_version,
            "method": "finite_difference",
            "dyadic_order": self.dyadic_order,
            "max_batch": self.max_batch,
            "n_jobs": 1,
            "dtype": "float64",
            "time_aug": False,
            "lead_lag": False,
            "normalize_gram": False,
            "static_kernel": self.kernel_parameters,
            "estimator": "unbiased_mmd_squared",
        }
