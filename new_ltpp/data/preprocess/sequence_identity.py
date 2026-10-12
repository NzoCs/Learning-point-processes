"""Stable sequence identity, independent of loader order and machine paths."""

import hashlib
import json


def sequence_ids(data, identifiers=None, *, global_ids=False):
    size = len(data["time_seqs"])
    if identifiers is not None and len(identifiers) != size:
        raise ValueError("Source identifiers must match the number of sequences")
    if identifiers is not None and global_ids:
        if any(value is None for value in identifiers):
            raise ValueError("Source sequence identifiers must not be missing")
        return [
            "provided:" + json.dumps(value, sort_keys=True) for value in identifiers
        ]
    digest = hashlib.sha256(b"sequence-snapshot-v1\n")
    for row in range(size):
        payload = [
            data[key][row] for key in ("time_seqs", "time_delta_seqs", "type_seqs")
        ]
        encoded = json.dumps(payload, allow_nan=False, default=lambda v: v.tolist())
        digest.update(encoded.encode() + b"\n")
    namespace = digest.hexdigest()
    rows = range(size) if identifiers is None else identifiers
    if any(value is None for value in rows):
        raise ValueError("Source sequence identifiers must not be missing")
    return [
        f"snapshot:{namespace}:" + json.dumps(value, sort_keys=True) for value in rows
    ]


def identify_loaded_data(payload, source):
    columns = source.column_names if hasattr(source, "column_names") else source.keys()
    if "sequence_id" in columns:
        payload["sequence_ids"] = sequence_ids(
            payload, source["sequence_id"], global_ids=True
        )
    elif "seq_idx" in columns:
        payload["sequence_ids"] = sequence_ids(payload, source["seq_idx"])
    else:
        payload["sequence_ids"] = sequence_ids(payload)
    return payload
