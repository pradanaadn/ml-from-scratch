import numpy as np


def mean_absolute_error(y_true, y_pred):
    try:
        distance = np.subtract(y_true, y_pred)
        square_distance = np.abs(distance)
        mae = np.mean(square_distance)

        return mae
    except Exception as e:
        print(e)
