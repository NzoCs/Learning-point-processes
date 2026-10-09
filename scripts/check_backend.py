"""Compute a tiny Gram with the real project backend on the requested device."""

import argparse
import json


def check_backend(device: str, backend: str = "pysiglib") -> dict:
    import torch
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.sig_kernel import (
        SIGKernel,
    )
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
        LinearKernel,
    )
    from new_ltpp.shared_types import Batch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but no CUDA device is available to this job."
        )
    times = torch.tensor([[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]], device=device)
    batch = Batch(
        time_seqs=times,
        time_delta_seqs=torch.diff(times, prepend=torch.zeros_like(times[:, :1])),
        type_seqs=torch.tensor([[0, 1, 0], [1, 0, 1]], device=device),
        valid_event_mask=torch.ones_like(times, dtype=torch.bool),
    )
    kernel = SIGKernel(
        static_kernel=LinearKernel(),
        embedding_type="linear",
        num_discretization_points=8,
        dyadic_order=0,
        num_event_types=2,
        backend=backend,
    )
    gram = kernel.compute_gram_matrix(batch, batch)
    if gram.shape != (2, 2) or not torch.isfinite(gram).all().item():
        raise RuntimeError("Signature Gram has invalid shape or non-finite values.")
    if gram.device.type != device:
        raise RuntimeError(f"Signature Gram is on {gram.device}, expected {device}.")
    if not torch.allclose(gram, gram.T, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            "Signature Gram is not symmetric within the smoke tolerance."
        )
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "backend": kernel.backend_name,
        "backend_details": kernel.kernel.metadata()
        if backend == "pysiglib"
        else {"reference_commit": "40a583155ea8d2194af0e90dddab37e2659cfcfd"},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(gram.device),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(gram.dtype),
        "gram": gram.detach().cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--backend", choices=("pysiglib", "sigkernel"), default="pysiglib"
    )
    args = parser.parse_args()
    print(json.dumps(check_backend(args.device, args.backend), indent=2))


if __name__ == "__main__":
    main()
