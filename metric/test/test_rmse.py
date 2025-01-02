from metric.implementation.RMSE import root_mean_squared_error


def test_rmse_scalar():
    y_true = 12
    y_pred = 10

    mse = root_mean_squared_error(y_true, y_pred)
    expected_result = 2
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"


def test_rmse_list():
    y_true = [1, 2, 3, 4, 5, 6]
    y_pred = [4, 5, 6, 7, 8, 9]

    mse = root_mean_squared_error(y_true, y_pred)
    expected_result = 3
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"
