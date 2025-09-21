"""Unit tests for gen_utils.py utility functions."""

import numpy as np
import pytest
import torch

from easy_tpp.utils import gen_utils


def test_generate_synthetic_data_basic():
    events = gen_utils.generate_synthetic_data(
        n_nodes=2, end_time=10, baseline=0.2, adjacency=0.1, decay=1.0
    )
    assert isinstance(events, list)
    assert len(events) == 2
    # At least one event per node (usually)
    assert all(isinstance(ev, list) for ev in events)
    assert all(isinstance(e, dict) for node in events for e in node)
    # Check event dict keys
    for node in events:
        for e in node:
            assert "time_since_start" in e
            assert "time_since_last_event" in e
            assert "type_event" in e


def test_format_multivariate_simulations():
    # Simulate a batch of 2 sequences
    simulations = [
        {
            "time_seq": torch.tensor([0.0, 1.0, 2.0]),
            "event_seq": torch.tensor([0, 1, 0]),
            "time_delta_seq": torch.tensor([0.0, 1.0, 1.0]),
        },
        {
            "time_seq": torch.tensor([0.0, 0.5]),
            "event_seq": torch.tensor([1, 0]),
            "time_delta_seq": torch.tensor([0.0, 0.5]),
        },
    ]
    formatted = gen_utils.format_multivariate_simulations(simulations, dim_process=2)
    assert isinstance(formatted, list)
    assert len(formatted) == 2
    for seq in formatted:
        assert "dim_process" in seq
        assert "seq_len" in seq
        assert "seq_idx" in seq
        assert "time_since_start" in seq
        assert "time_since_last_event" in seq
        assert "type_event" in seq


def test_format_tick_data_to_hf():
    # Create synthetic event data
    events = [
        [
            {"time_since_start": 0.0, "time_since_last_event": 0.0, "type_event": 0},
            {"time_since_start": 1.0, "time_since_last_event": 1.0, "type_event": 0},
        ],
        [{"time_since_start": 0.5, "time_since_last_event": 0.5, "type_event": 1}],
    ]
    formatted = gen_utils.format_tick_data_to_hf(events, dim_process=2, max_seq_len=2)
    assert isinstance(formatted, list)
    assert all("dim_process" in seq for seq in formatted)
    assert all("seq_len" in seq for seq in formatted)
    assert all("seq_idx" in seq for seq in formatted)
    assert all("time_since_start" in seq for seq in formatted)
    assert all("type_event" in seq for seq in formatted)
