from metric.implementation.MAE import mean_absolute_error


def test_mae_scalar():
    y_true = 12
    y_pred = 10

    mse = mean_absolute_error(y_true, y_pred)
    expected_result = 2
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"

def test_mae_scalar_negative():
    y_true = 12
    y_pred = 14

    mse = mean_absolute_error(y_true, y_pred)
    expected_result = 2
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"
    
def test_mae_list_int():
    y_true = [1, 2, 3, 4, 5, 6]
    y_pred = [4, 5, 6, 7, 8, 9]

    mse = mean_absolute_error(y_true, y_pred)
    expected_result = 3
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"
def test_mae_list_float():
    y_true = [1.08, 1.2, 1.4, 2.1, 1.9, 7, 2.9]
    y_pred = [0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1]

    mse = round(mean_absolute_error(y_true, y_pred), 3)
    expected_result = 0.811
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"