import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between the true and predicted values.

    MSE is a measure of the average squared difference between the estimated values
    and the actual value. It is used to evaluate the accuracy of a model.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The mean squared error between the true and predicted values.

    Raises:
    Exception: If an error occurs during the calculation.
    """
    try:
        distance = np.subtract(y_true, y_pred)
        square_distance = np.power(distance, 2)
        mse = np.mean(square_distance)

        return mse
    except Exception as e:
        print(e)
