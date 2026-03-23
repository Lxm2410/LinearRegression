import numpy as np
import math


class GDModel:
    def __init__(self, num_samples, num_features):
        rng = np.random.default_rng()
        self.X = np.array([])
        self.beta = rng.random(num_features + 1) # one constant term
        self.num_samples = num_samples
        self.grad = np.array([])
        self.loss_grad = np.array([])
    
    def forward(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate([ones, X], axis=1)
        self.X = X
        ans = np.dot(X, self.beta)
        self.loss_components = np.dot(X, self.beta)
        return ans


    def backward(self):
        self.grad = np.dot(self.X.T, self.loss_grad)
    
    def fit(self, X_train, Y_train, lr):
        e = math.inf
        cnt = 0
        last = 0
        while e > 1e-4:
            cnt += 1
            Y_hat = self.forward(X_train)
            l = self.loss(Y_hat, Y_train)
            e = abs(l - last)
            last = l
            self.backward()
            if cnt % 4 == 0:
                lr /= 2
            self.beta -= lr * self.grad


    def loss(self, Y, labels):
        self.loss_grad = 2 / self.num_samples * (Y - labels)
        return np.mean((Y - labels) ** 2)
    
    def evaluate(self, X_test, labels):
        prediction = self.forward(X_test)
        return np.mean(self.loss(prediction, labels))