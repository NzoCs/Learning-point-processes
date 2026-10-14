"""Exactness controls for opt-in global signature calibration."""

import numpy as np
import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import heterogeneous_histories, swap_orders
from scripts.global_signature_calibration import (
    blocks,
    compressed_encoding,
    delta_statistics,
    signature_delta,
    verify_public_statistics,
)


def kernel():
    return SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2)


def future(n=32, shifted=False):
    t = torch.tensor([[0.1, 0.6], [0.0, 0.0], [0.0, 0.2], [0.1, 0.6]]).repeat(n // 4, 1)
    m = torch.tensor([[0, 1], [2, 2], [2, 1], [0, 1]]).repeat(n // 4, 1)
    mask = torch.tensor(
        [[True, True], [False, False], [False, True], [True, True]]
    ).repeat(n // 4, 1)
    if shifted:
        t = t * 2
    gaps = torch.diff(t, prepend=torch.zeros(n, 1)).masked_fill(~mask, 0)
    return Batch(t, gaps, m, mask)


def test_duplicate_compression_preserves_gram_multiplicity_padding_and_inputs():
    batch = future()
    before = batch.time_seqs.clone()
    sig = kernel()
    kg, index = compressed_encoding(sig, batch)
    assert len(kg) == 3 and len(index) == 32
    np.testing.assert_allclose(
        kg[np.ix_(index, index)],
        sig.compute_gram_matrix(batch, batch).numpy(),
        rtol=1e-12,
        atol=1e-13,
    )
    assert torch.equal(batch.time_seqs, before)
    assert batch.no_next_event_mask is None


def test_global_delta_and_every_swap_match_public_joint_kernel():
    h, x, y = heterogeneous_histories(32), future(), future(shifted=True)
    orders = swap_orders(32, 99, 781)
    signs = 1.0 - 2 * (orders[:, :32].numpy() // 32)
    sig = kernel()
    delta, meta = signature_delta(sig, h, x, y, 2.0)
    assert meta["unique_future_paths"] < meta["total_future_paths"]
    verify_public_statistics(sig, h, x, y, delta, signs, 2.0)
    observed = delta_statistics(delta, signs)
    # A uniform full pair swap negates all signs and leaves the two-sided MMD unchanged.
    np.testing.assert_allclose(
        observed, delta_statistics(delta, -signs), rtol=1e-12, atol=1e-13
    )


def test_full_window_and_batch_contracts_are_enforced():
    h, x, y = heterogeneous_histories(32), future(), future()
    x.time_seqs[0, 0] = 2.1
    with pytest.raises(ValueError, match="fixed horizon"):
        signature_delta(kernel(), h, x, y, 2.0)
    x.time_seqs[0, 0] = 0.1
    x.no_next_event_mask = torch.ones(32, dtype=torch.bool)
    with pytest.raises(ValueError, match="permanent termination"):
        signature_delta(kernel(), h, x, y, 2.0)
    with pytest.raises(ValueError, match="16/11/5"):
        list(blocks(33))
    assert list(blocks(32)) == [(0, 16), (16, 27), (27, 32)]
