import numpy as np

def hinge_loss(y_true, y_pred):
    """
    Calculate the hinge loss for binary or multiclass classification.
    
    Parameters:
    y_true (array-like): True labels, expected to be in {0, 1} for binary or {0, 1, ..., num_classes-1} for multiclass
    y_pred (array-like): Predicted labels, can be any real number for binary or predicted scores for each class for multiclass
    
    Returns:
    float: Hinge loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    is_multiclass = len(y_pred.shape) > 1 and y_pred.shape[1] > 1
    
    if not is_multiclass:
        # Binary classification: Map y_true from {0, 1} to {-1, 1}
        y_true = np.where(y_true == 0, -1, 1)
        
        # Calculate hinge loss
        loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
    else:
        # Multiclass classification
        num_samples = y_true.shape[0]
        
        # Create a mask for the correct class scores
        correct_class_scores = y_pred[np.arange(num_samples), y_true]
        
        # Calculate the hinge loss
        margins = np.maximum(0, y_pred - correct_class_scores[:, np.newaxis] + 1)
        margins[np.arange(num_samples), y_true] = 0  # Do not consider correct class in the loss
        loss = np.sum(margins) / num_samples
    
    return loss