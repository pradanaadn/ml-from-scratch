from metric.implementation.Hinge import hinge_loss
import pytest
import numpy as np

def test_hinge_binary():
    # Test case 1: Perfect prediction
    y_true = [1, 0, 1, 0]
    y_pred = [1, -1, 1, -1]
    expected_loss = 0.0
    assert hinge_loss(y_true, y_pred) == pytest.approx(expected_loss, 0.001)
    
    # Test case 2: Some incorrect predictions
    y_true = [1, 0, 1, 0]
    y_pred = [0.5, -0.5, -0.5, 0.5]
    expected_loss = 1.0
    assert hinge_loss(y_true, y_pred) == pytest.approx(expected_loss, 0.001)
    
    # Test case 3: All incorrect predictions
    y_true = [1, 0, 1, 0]
    y_pred = [-1, 1, -1, 1]
    expected_loss = 2.0
    assert hinge_loss(y_true, y_pred) == pytest.approx(expected_loss, 0.001)

def test_hinge_multiclass():
    # Test case 1: Perfect prediction
    y_true = [0, 1, 2]
    y_pred = [[3, 1, 0], [1, 3, 0], [0, 1, 3]]
    expected_loss = 0.0
    assert hinge_loss(y_true, y_pred) == pytest.approx(expected_loss, 0.001)
    
    # Test case 2: Some incorrect predictions
    y_true = [0, 1, 2]
    y_pred = [[1, 3, 0], [3, 1, 0], [0, 3, 1]]
    expected_loss = 3.0  # Corrected expected loss
    assert hinge_loss(y_true, y_pred) == pytest.approx(expected_loss, 0.001)
    
    # Test case 3: All incorrect predictions
    y_true = [0, 1, 2]
    y_pred = [[0, 3, 1], [0, 1, 3], [3, 0, 1]]
    expected_loss = 4.0
    assert hinge_loss(y_true, y_pred) == pytest.approx(expected_loss, 0.001)
