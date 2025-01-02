import numpy as np


def root_mean_squared_error(y_true, y_pred):
    try:
        distance = np.subtract(y_true, y_pred)
        square_distance = np.power(distance, 2)
        mse = np.mean(square_distance)
        rmse = np.sqrt(mse)
        return rmse
    except Exception as e:
        print(e)
