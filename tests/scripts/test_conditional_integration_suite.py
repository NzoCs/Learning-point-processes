"""The optional conditional suite must use reproducible, disjoint source datasets."""

from scripts.conditional_integration_suite import make_splits


def test_independent_train_validation_and_evaluation_fixtures():
    splits = make_splits()
    assert splits == make_splits()
    assert set(splits) == {"train", "dev", "test"}
    identifiers = []
    trajectories = []
    for records in splits.values():
        assert len(records) == 8
        for row in records:
            identifiers.append(row["sequence_id"])
            trajectories.append(tuple(row["time_since_start"]))
            assert len(row["type_event"]) == len(row["time_since_start"]) == 5
            assert row["time_since_last_event"][0] == 0
            assert all(gap > 0 for gap in row["time_since_last_event"][1:])
            assert all(mark in (0, 1) for mark in row["type_event"])
    assert len(set(identifiers)) == len(set(trajectories)) == 24
