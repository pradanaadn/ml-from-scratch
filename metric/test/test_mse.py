from metric.implementation.MSE import mean_squared_error

def test_mse_scalar():
    y_true = 12
    y_pred = 10
    
    mse = mean_squared_error(y_true, y_pred)
    expected_result = 4
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"

def test_mse_list():
    y_true = [1,2,3,4,5,6]
    y_pred = [4,5,6,7,8,9]
    
    mse = mean_squared_error(y_true, y_pred)
    expected_result = 9
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"
   