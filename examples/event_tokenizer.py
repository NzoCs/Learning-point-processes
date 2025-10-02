#!/usr/bin/env python3
import random
from typing import Any, Dict, List

from easy_tpp.configs import TokenizerConfig
from easy_tpp.data.preprocess import EventTokenizer


def make_raw_data() -> List[List[Dict[str, Any]]]:
    data = [
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 0}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
        [{"time_since_last_event": 0, "time_since_start": 0, "type_event": 1}],
    ]
    for i, j in enumerate([2, 5, 3]):
        start_time = 0
        for k in range(j):
            delta_t = random.random()
            start_time += delta_t
            data[i].append(
                {
                    "time_since_last_event": delta_t,
                    "time_since_start": start_time,
                    "type_event": random.randint(0, 10),
                }
            )
    return data


def main() -> None:
    source_data = make_raw_data()

    time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
    type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
    time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]

    input_data = {
        "time_seqs": time_seqs,
        "type_seqs": type_seqs,
        "time_delta_seqs": time_delta_seqs,
    }

    # Use modern API
    config = TokenizerConfig(num_event_types=11, pad_token_id=11)

    tokenizer = EventTokenizer(config)

    # Tokenize and pad data
    output = tokenizer.pad(input_data, return_tensors="pt")

    print("Tokenized output:")
    print(f"Time sequences shape: {output['time_seqs'].shape}")
    print(f"Type sequences shape: {output['type_seqs'].shape}")
    print(f"Time delta sequences shape: {output['time_delta_seqs'].shape}")


if __name__ == "__main__":
    main()
