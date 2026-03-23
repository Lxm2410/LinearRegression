
import numpy as np

class AnalyticLinearReg:

    def __init__(self):
        self.beta = np.array([])
    
    def predict(self, X):
        return np.dot(X, self.beta)
    
    def loss(self, Y, Y_hat):
        ''' returns mean squared error of prediction
        '''
        return np.mean((Y - Y_hat) ** 2)

    def evaluate(self, X_test, labels):
        prediction = self.predict(X_test)
        return self.loss(labels, prediction)
    
    def fit(self, X_train, labels):
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), labels)
    
    def __apply__(self, X):
        return self.predict(X)

