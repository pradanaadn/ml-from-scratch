import pytest
import numpy as np
from model.implementation.regression import LinearRegression

def test_linear_regression_fit():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = LinearRegression()
    model.fit(X, y)
    assert model.weights is not None

def test_linear_regression_predict():
    X_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(X_train, np.array([1, 2])) + 3
    X_test = np.array([[3, 5], [5, 9]])
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    expected_predictions = np.dot(np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1), model.weights)
    np.testing.assert_array_almost_equal(predictions, expected_predictions)

def test_linear_regression_perfect_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_array_almost_equal(predictions, y)



def test_linear_regression_single_feature():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 3, 4, 5])
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_array_almost_equal(predictions, y)