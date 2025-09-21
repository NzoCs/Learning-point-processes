from typing import Any, Dict

import yaml


def parse_runner_yaml_config(
    yaml_path: str, experiment_id: str, dataset_id: str
) -> Dict[str, Any]:
    """
    Parse the runner YAML config and extract the correct sub-configs for RunnerConfig.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Get data config for the dataset
    data_section = config.get("data", {})
    if dataset_id not in data_section:
        raise ValueError(f"Dataset id '{dataset_id}' not found in YAML data section.")
    data_config = data_section[dataset_id]
    data_config["dataset_id"] = dataset_id

    # 2. Get experiment section
    if experiment_id not in config:
        raise ValueError(f"Experiment id '{experiment_id}' not found in YAML.")
    exp_section = config[experiment_id]  # 3. Model config
    model_config = exp_section.get("model_config", {})
    model_config["model_id"] = model_config.get("model_id", experiment_id)

    # Extract num_event_types from data_specs and add to model_config
    data_specs = data_config.get("data_specs", {})
    if "num_event_types" in data_specs:
        model_config["num_event_types"] = data_specs["num_event_types"]
    else:
        raise ValueError(
            f"num_event_types not found in data_specs for dataset '{dataset_id}'"
        )

    # 4. Trainer config (optional, may not be present)
    trainer_config = exp_section.get("trainer_config", {})
    # Add batch_size from data_loading_specs if not present
    data_loading_specs = exp_section.get("data_loading_specs", {})
    if "batch_size" in data_loading_specs and "batch_size" not in trainer_config:
        trainer_config["batch_size"] = data_loading_specs["batch_size"]
    # Add dataset_id and model_id
    trainer_config["dataset_id"] = dataset_id
    trainer_config["model_id"] = model_config["model_id"]

    # 5. Compose the final config dict
    return {
        "trainer_config": trainer_config,
        "model_config": model_config,
        "data_config": data_config,
    }
