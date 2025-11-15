import logging
import re
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def flexible_state_dict_loading(
    model: torch.nn.Module, state_dict: Dict[str, Any], strict: bool = False
):
    """
    Load a state dictionary with flexible handling of size mismatches.

    This function attempts to load as much as possible from a state dict even when
    there are size mismatches between the saved weights and the model parameters.

    Args:
        model: The model to load the state dict into
        state_dict: The state dict containing the weights
        strict: If True, raises an error for any missing or unexpected keys
               If False, allows partial loading and warns about mismatches

    Returns:
        dict: A report of the loading process with keys loaded, missing, unexpected, and shape_mismatch
    """
    model_state_dict = model.state_dict()

    # Keep track of loading stats
    keys_loaded = []
    shape_mismatch = {}
    missing_keys = []
    unexpected_keys = []

    # Create a new state dict with compatible weights
    compatible_state_dict = {}

    # First, identify keys that don't match in shape
    for key, param in model_state_dict.items():
        if key in state_dict:
            if param.shape == state_dict[key].shape:
                compatible_state_dict[key] = state_dict[key]
                keys_loaded.append(key)
            else:
                shape_mismatch[key] = {
                    "model_shape": list(param.shape),
                    "checkpoint_shape": list(state_dict[key].shape),
                }
        else:
            missing_keys.append(key)

    # Identify keys in state_dict that don't exist in model
    for key in state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)

    # Load the compatible parts
    try:
        model.load_state_dict(compatible_state_dict, strict=False)
    except RuntimeError as e:
        # If there are still shape mismatches, skip those keys and try again
        error_msg = str(e)
        import re

        # Find all keys with size mismatch
        mismatched_keys = re.findall(r"size mismatch for ([^:]+):", error_msg)
        for key in mismatched_keys:
            if key in compatible_state_dict:
                logger.warning(
                    f"Skipping incompatible key {key} due to shape mismatch."
                )
                del compatible_state_dict[key]
        # Try loading again with problematic keys removed
        model.load_state_dict(compatible_state_dict, strict=False)

    # Report on loading
    report = {
        "keys_loaded": keys_loaded,
        "shape_mismatch": shape_mismatch,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }

    if strict and (missing_keys or unexpected_keys):
        error_msg = f"Error(s) in loading state_dict:\n"
        if missing_keys:
            error_msg += f"Missing keys: {missing_keys}\n"
        if unexpected_keys:
            error_msg += f"Unexpected keys: {unexpected_keys}\n"
        if shape_mismatch:
            error_msg += f"Shape mismatches: {shape_mismatch}\n"
        raise RuntimeError(error_msg)

    return report


def compare_model_configs(model_config, checkpoint_config):
    """
    Compare the model configuration with the configuration from a checkpoint
    to identify potential issues before loading.

    Args:
        model_config: The configuration of the current model
        checkpoint_config: The configuration from the checkpoint

    Returns:
        dict: A report of differences between configurations
    """
    if checkpoint_config is None:
        return {"error": "No checkpoint configuration available"}

    differences = {}
    warnings = []

    # Convert configs to dictionaries if they're not already
    if not isinstance(model_config, dict):
        model_config = (
            vars(model_config) if hasattr(model_config, "__dict__") else model_config
        )

    if not isinstance(checkpoint_config, dict):
        checkpoint_config = (
            vars(checkpoint_config)
            if hasattr(checkpoint_config, "__dict__")
            else checkpoint_config
        )

    # Get all keys from both configs
    all_keys = set(model_config.keys()) | set(checkpoint_config.keys())

    # Compare values
    for key in all_keys:
        if key not in model_config:
            differences[key] = {
                "status": "missing_in_model",
                "checkpoint_value": checkpoint_config[key],
            }
            warnings.append(f"Config key '{key}' exists in checkpoint but not in model")
        elif key not in checkpoint_config:
            differences[key] = {
                "status": "missing_in_checkpoint",
                "model_value": model_config[key],
            }
        elif model_config[key] != checkpoint_config[key]:
            differences[key] = {
                "status": "different",
                "model_value": model_config[key],
                "checkpoint_value": checkpoint_config[key],
            }

            # Add specific warnings for keys that might cause dimension mismatch
            if key in [
                "hidden_size",
                "embedding_dim",
                "n_layers",
                "n_heads",
                "num_types",
                "event_type_categories",
            ]:
                warnings.append(
                    f"Potential dimension mismatch: {key} differs - model: {model_config[key]}, checkpoint: {checkpoint_config[key]}"
                )

    return {"differences": differences, "warnings": warnings}
