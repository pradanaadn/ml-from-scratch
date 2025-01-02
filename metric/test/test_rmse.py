from metric.implementation.RMSE import root_mean_squared_error


def test_rmse_scalar():
    y_true = 12
    y_pred = 10

    mse = root_mean_squared_error(y_true, y_pred)
    expected_result = 2
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"


def test_rmse_list_int():
    y_true = [1, 2, 3, 4, 5, 6]
    y_pred = [4, 5, 6, 7, 8, 9]

    mse = root_mean_squared_error(y_true, y_pred)
    expected_result = 3
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"

def test_rmse_list_float():
    y_true = [1.08, 1.2, 1.4, 2.1, 1.9, 7, 2.9]
    y_pred = [0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1]

    mse = round(root_mean_squared_error(y_true, y_pred), 3)
    expected_result = 1.643
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"