#!/usr/bin/env python3
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import DataConfig
from easy_tpp.data.preprocess import TPPDataModule


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
            data[i].append({"time_since_last_event": delta_t,
                            "time_since_start": start_time,
                            "type_event": random.randint(0, 10)
                            })
    return data


def main() -> None:
    # Create data config
    data_config = DataConfig(
        dataset_id="example_data",
        data_format="dict",
        num_event_types=11,
        pad_token_id=11,
        batch_size=2
    )
    
    # Create data module
    datamodule = TPPDataModule(data_config)
    
    # Setup data module
    datamodule.setup(stage='fit')
    
    # Get data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    print("Train loader created successfully")
    print(f"Number of batches in train loader: {len(train_loader)}")


if __name__ == '__main__':
    main()
