"""Origin validation must distinguish rounded clocks from incorrect gaps."""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from new_ltpp.shared_types import Batch

# Literal first failing rows from Linux run 38714040860, replayed locally on
# the original seeds. These values are observations, not a duplicate oracle.
CASES = [
    (
        1.6147232055664062,
        [
            46.354774475097656,
            56.71011734008789,
            150.49368286132812,
            176.30816650390625,
            176.32827758789062,
            291.96697998046875,
        ],
        [
            46.354774475097656,
            10.355342864990234,
            93.78355407714844,
            25.814498901367188,
            0.0200958251953125,
            115.63870239257812,
        ],
    ),
    (
        0.3558502495288849,
        [0.374546617269516, 1006.02197265625, 1006.0857543945312],
        [0.374546617269516, 1005.6473999023438, 0.06375251710414886],
    ),
    (
        2.62347674369812,
        [
            3.872267007827759,
            4.587836265563965,
            515.24951171875,
            515.955322265625,
            516.0836181640625,
            516.4100952148438,
        ],
        [
            3.872267007827759,
            0.7155693173408508,
            510.66168212890625,
            0.7058391571044922,
            0.12830229103565216,
            0.3264908194541931,
        ],
    ),
]


def batches(origin, times, gaps, dtype=torch.float32):
    t, d = torch.tensor([times], dtype=dtype), torch.tensor([gaps], dtype=dtype)
    h = torch.tensor([[origin]], dtype=dtype)
    return Batch(h, h.clone(), h.long() * 0, h.bool() | True), Batch(
        t, d, t.long() * 0, t.bool() | True
    )


def make_test():
    return ConditionalMMDTwoSampleTest(
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), 9
    )


@pytest.mark.parametrize(
    "case", CASES, ids=["Hawkes-q6", "IntensityFree-q3", "IntensityFree-q6"]
)
def test_real_float32_clock_rows_are_valid_without_changing_data_or_rng(case):
    h, future = batches(*case)
    t, d = future.time_seqs.clone(), future.time_delta_seqs.clone()
    old_expected = torch.diff(t, prepend=torch.zeros_like(t[:, :1]), dim=1)
    assert not torch.allclose(d, old_expected, rtol=1e-5, atol=1e-5)
    rng = torch.get_rng_state().clone()
    make_test().validate_pairs(future, future, h)
    assert torch.equal(t, future.time_seqs) and torch.equal(d, future.time_delta_seqs)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("case", CASES)
def test_dtype_controls_roundoff_allowance_and_real_errors_are_rejected(case):
    # Declaring these rounded observations as float64 cannot justify float32
    # rounding. A new true float64 simulation must honor its higher precision.
    h, future = batches(*case, dtype=torch.float64)
    with pytest.raises(ValueError, match="origin"):
        make_test().validate_pairs(future, future, h)
    h, future = batches(*case)
    future.time_delta_seqs[0, -1] += 0.01
    with pytest.raises(ValueError, match="origin"):
        make_test().validate_pairs(future, future, h)


@pytest.mark.parametrize("bad", [-1e-7, float("nan"), float("inf"), -float("inf")])
def test_invalid_sampled_gaps_are_not_hidden_by_a_precision_budget(bad):
    h, future = batches(2.0, [0.0, 1006.0], [bad, 1006.0])
    with pytest.raises(ValueError, match="origin"):
        make_test().validate_pairs(future, future, h)


def test_history_origin_and_masked_padding_are_handled_explicitly():
    # At a large source clock the first short gap is rounded in the returned
    # relative time too. The bound needs the actual history endpoint.
    origin = torch.tensor(4096.0)
    delay = torch.tensor(0.0201)
    relative = (origin + delay) - origin
    h, future = batches(float(origin), [float(relative)], [float(delay)])
    h.time_seqs = torch.tensor([[origin, float("nan")]])
    h.time_delta_seqs = torch.tensor([[origin, 0.0]])
    h.type_seqs = torch.tensor([[0, 2]])
    h.valid_event_mask = torch.tensor([[True, False]])
    make_test().validate_pairs(future, future, h)
    h.time_seqs[0, 0] = 0.0
    with pytest.raises(ValueError, match="origin"):
        make_test().validate_pairs(future, future, h)


def test_actual_wrong_time_origin_is_still_rejected_at_late_events():
    h, future = batches(2.0, [1006.0, 1006.1], [1004.0, 0.1])
    with pytest.raises(ValueError, match="origin"):
        make_test().validate_pairs(future, future, h)
