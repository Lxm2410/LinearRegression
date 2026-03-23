import numpy as np
import math


class GDModel:
    def __init__(self, num_features):
        self.rng = np.random.default_rng(12345)
        self.X = np.array([])
        self.beta = self.rng.random(num_features)
        self.grad = np.array([])
        self.loss_grad = np.array(num_features)
    
    def forward(self, X):
        self.X = X
        ans = np.dot(X, self.beta)
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
            self.beta -= lr * self.grad
#            if cnt % 50 == 0:
#                print(f'Iteration {cnt} Loss: {l:.4f}')
        return cnt


    def loss(self, Y, labels):
        self.loss_grad = 2 / Y.shape[0] * (Y - labels)
        return np.mean((Y - labels) ** 2)
    
    def evaluate(self, X_test, labels):
        prediction = self.forward(X_test)
        return np.mean(self.loss(prediction, labels))