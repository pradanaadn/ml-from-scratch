import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X @ self.weights