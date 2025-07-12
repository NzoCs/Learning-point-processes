import torch
from typing import Union, Tuple, Any


def ensure_same_device(
    *tensors, target_device: Union[str, torch.device] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Ensure all tensors are on the same device.

    Args:
        *tensors: Variable number of tensors or tensor-like objects
        target_device: Target device. If None, uses the device of the first tensor

    Returns:
        Tuple of tensors all on the same device
    """
    if not tensors:
        return ()

    # Filter out None values and non-tensors
    valid_tensors = []
    tensor_indices = []
    for i, tensor in enumerate(tensors):
        if tensor is not None and isinstance(tensor, torch.Tensor):
            valid_tensors.append(tensor)
            tensor_indices.append(i)

    if not valid_tensors:
        return tensors

    # Determine target device
    if target_device is None:
        target_device = valid_tensors[0].device
    elif isinstance(target_device, str):
        target_device = torch.device(target_device)

    # Convert all tensors to target device
    result = list(tensors)
    for i in tensor_indices:
        if result[i] is not None:
            result[i] = result[i].to(target_device)

    return tuple(result) if len(result) > 1 else result[0]


def get_device_info():
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "current_device": None,
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()

    return info


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
