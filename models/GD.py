import numpy as np


class GDModel:
    def __init__(self):
        self.beta = np.array([])
        self.grad = np.array([])
    
    def forward(self, X):
        ones = np.ones((X.shape[0]))
        X = np.concatenate([ones, X], axis=1)
        return np.dot(X, self.beta)