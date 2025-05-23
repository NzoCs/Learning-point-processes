import torch

def ensure_same_device(*tensors, target_device=None):
    """
    Ensures that all given tensors are on the same device.
    If target_device is provided, all tensors will be moved to that device.
    Otherwise, they will be moved to the device of the first tensor.
    
    Args:
        *tensors: Variable number of tensors to ensure are on the same device
        target_device: Optional target device to move tensors to
        
    Returns:
        List of tensors all on the same device
    """
    if not tensors:
        return []
    
    # Determine target device
    if target_device is None:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                target_device = t.device
                break
        if target_device is None:
            # No tensor found, return as is
            return list(tensors)
    
    # Move all tensors to target device
    result = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.device != target_device:
            result.append(t.to(target_device))
        else:
            result.append(t)
    
    return result
