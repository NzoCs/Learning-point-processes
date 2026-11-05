"""Evaluate the public test API on one frozen independent trajectory pool.

Generation, train/validation/test splitting and observation-window selection
precede this function. Never count real-data rejections as calibrated power.
"""

import hashlib
import time

import torch

from new_ltpp.configs.window_test_config import WindowTestConfig
from new_ltpp.evaluation.statistical_testing.observations import configurations
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    create_statistical_test,
)


def batch_digest(batch):
    digest = hashlib.sha256()
    for value in [batch.time_seqs, batch.type_seqs, batch.valid_event_mask]:
        a = value.detach().cpu().contiguous().numpy()
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def evaluate_test_suite(observed, simulated, reference, configs):
    """Use the same pools for every test; return raw null vectors and timing.

    Model-independent MMD gets two samples. Relative loss gets both samples
    and the common reference. Stein gets observed data and the fixed model.
    These null hypotheses remain distinct in returned metadata.
    """
    if len(observed.time_seqs) != len(simulated.time_seqs):
        raise ValueError("Matched independent pools required")
    configs = [
        (
            raw
            if isinstance(raw, WindowTestConfig)
            else WindowTestConfig.model_validate(raw)
        )
        for raw in configs
    ]
    if not configs or len({(c.horizon, c.num_event_types) for c in configs}) != 1:
        raise ValueError("All tests must share one exposure and mark alphabet")
    if observed.time_seqs.device != simulated.time_seqs.device:
        raise ValueError("Pools must use the same device")
    output = []
    for config in configs:
        configurations(observed, config.horizon, config.num_event_types)
        configurations(simulated, config.horizon, config.num_event_types)
        test = create_statistical_test(config, reference=reference)
        if observed.time_seqs.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(observed.time_seqs.device)
        started = time.perf_counter()
        result = (
            test.compute_statistics(observed)
            if config.test_type == "ksd"
            else test.compute_statistics(observed, simulated)
        )
        if observed.time_seqs.device.type == "cuda":
            torch.cuda.synchronize()
        output.append(
            dict(
                test=test.name,
                config=config.model_dump(),
                null_hypothesis=(
                    "observed_equals_fixed_model"
                    if config.test_type == "ksd"
                    else "exchangeable_observed_simulated_pairs"
                ),
                elapsed_seconds=time.perf_counter() - started,
                peak_gpu_memory_bytes=(
                    torch.cuda.max_memory_allocated(observed.time_seqs.device)
                    if observed.time_seqs.device.type == "cuda"
                    else None
                ),
                **{
                    k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in result.items()
                },
            )
        )
    return dict(
        n=len(observed.time_seqs),
        observed_sha256=batch_digest(observed),
        simulated_sha256=batch_digest(simulated),
        tests=output,
    )
