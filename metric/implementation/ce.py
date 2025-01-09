import numpy as np 


def cross_entropy(y_true, y_pred):
    """
    Computes the cross-entropy loss between true labels and predicted probabilities.
    Cross-entropy loss is a measure of the difference between two probability distributions
    for a given random variable or set of events. It is commonly used in classification tasks.
    Parameters:
    y_true (numpy.ndarray): Array of true labels (one-hot encoded).
    y_pred (numpy.ndarray): Array of predicted probabilities.
    Returns:
    float: The cross-entropy loss.
    """
    
    log_y_pred = np.log10(y_pred)
    y_true_x_y_pred = np.multiply(y_true, log_y_pred)
    loss = - np.sum(y_true_x_y_pred)
    return loss