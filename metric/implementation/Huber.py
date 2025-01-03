import numpy as np


def huber_loss(y_true, y_pred, delta):
    """
    Calculate the Huber loss, which is less sensitive to outliers in data than the squared error loss.
    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    delta (float): The threshold at which to switch between mean squared error and mean absolute error.
    Returns:
    float: The Huber loss value.
    """
    difference = np.subtract(y_true, y_pred)
    defference_square = np.square(difference)
    half_delta = np.divide(delta, 2)
    difference_abs = np.abs(difference)
    difference_abs_half_delta = np.subtract(difference_abs, half_delta)

    huber_mse = np.divide(defference_square, 2)
    huber_mae = np.multiply(difference_abs_half_delta, delta)
    huber_conditional = np.where(difference <= delta, huber_mse, huber_mae)
    huber_loss = np.mean(huber_conditional)
    
    return huber_loss
