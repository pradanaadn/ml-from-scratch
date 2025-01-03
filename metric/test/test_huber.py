from metric.implementation.Huber import huber_loss


def test_mse_list_float():
    y_true = [1.08, 1.2, 1.4, 2.1, 1.9, 7, 2.9]
    y_pred = [0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1]
    
    mse = round(huber_loss(y_true, y_pred, 1.35), 3)
    expected_result = 0.728
    assert expected_result == mse, f"Expected {expected_result}, but got {mse}"

def test_huber_loss_zero_difference():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    
    loss = huber_loss(y_true, y_pred, 1.0)
    expected_result = 0.0
    assert expected_result == loss, f"Expected {expected_result}, but got {loss}"

def test_huber_loss_large_delta():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 2.5, 3.5]
    
    loss = round(huber_loss(y_true, y_pred, 10.0), 3)
    expected_result = 0.125
    assert expected_result == loss, f"Expected {expected_result}, but got {loss}"

def test_huber_loss_small_delta():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 2.5, 3.5]
    
    loss = round(huber_loss(y_true, y_pred, 0.1), 3)
    expected_result = 0.045
    assert expected_result == loss, f"Expected {expected_result}, but got {loss}"

def test_huber_loss_negative_values():
    y_true = [-1.0, -2.0, -3.0]
    y_pred = [-1.5, -2.5, -3.5]
    
    loss = round(huber_loss(y_true, y_pred, 1), 3)
    expected_result = 0.125
    assert expected_result == loss, f"Expected {expected_result}, but got {loss}"