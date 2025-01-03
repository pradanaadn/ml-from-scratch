import numpy as np


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    MAE is the average of the absolute differences between the true values and the predicted values.
    It is a measure of the accuracy of a predictive model.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The mean absolute error.

    Raises:
    Exception: If an error occurs during the calculation.
    """
    try:
        distance = np.subtract(y_true, y_pred)
        absolute_distance = np.abs(distance)
        mae = np.mean(absolute_distance)

        return mae
    except Exception as e:
        print(e)
