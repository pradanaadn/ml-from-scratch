import numpy as np


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the true and predicted values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The RMSE value.

    Raises:
    Exception: If an error occurs during the calculation.
    """
    try:
        distance = np.subtract(y_true, y_pred)
        square_distance = np.power(distance, 2)
        mse = np.mean(square_distance)
        rmse = np.sqrt(mse)
        return rmse
    except Exception as e:
        print(e)
