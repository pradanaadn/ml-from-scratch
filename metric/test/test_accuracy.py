import pytest
from metric.implementation.accuracy import accuracy

def test_accuracy_binary():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    assert accuracy(y_true, y_pred) == 0.8

def test_accuracy_multiclass():
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 2, 2, 1, 0]
    assert accuracy(y_true, y_pred) == 0.8

def test_accuracy_all_correct():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    assert accuracy(y_true, y_pred) == 1.0

def test_accuracy_all_incorrect():
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    assert accuracy(y_true, y_pred) == 0.0

def test_accuracy_empty():
    y_true = []
    y_pred = []
    assert accuracy(y_true, y_pred) == 0.0

def test_accuracy_length_mismatch():
    y_true = [0, 1]
    y_pred = [0, 1, 0]
    with pytest.raises(ValueError):
        accuracy(y_true, y_pred)