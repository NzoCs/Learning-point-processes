import numpy as np

from easy_tpp.utils.const import PredOutputIndex
from easy_tpp.utils.metrics import MetricsHelper


@MetricsHelper.register(name='rmse', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions using vectorized operations.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    
    # Extract the relevant predictions and labels using the mask
    pred = predictions[PredOutputIndex.TimePredIndex][seq_mask]
    label = labels[PredOutputIndex.TimePredIndex][seq_mask]

    # Use vectorized flatten instead of reshape for better performance
    pred = pred.flatten()
    label = label.flatten()
    
    # Compute difference vectorized, then square, mean, and sqrt
    diff_squared = np.square(pred - label)
    return np.sqrt(np.mean(diff_squared))


@MetricsHelper.register(name='acc', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def acc_metric_function(predictions, labels, **kwargs):
    """Compute accuracy ratio metrics of the type predictions using vectorized operations.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: accuracy ratio of the type predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    
    # Extract the relevant predictions and labels using the mask
    pred = predictions[PredOutputIndex.TypePredIndex][seq_mask]
    label = labels[PredOutputIndex.TypePredIndex][seq_mask]
    
    # Use flatten instead of reshape for better performance
    pred = pred.flatten()
    label = label.flatten()
    
    # Use vectorized equality comparison and mean calculation
    return np.mean(pred == label)
