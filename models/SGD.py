import numpy as np
from GradDescent import GDModel


class SGDModel:
    def __init__(self, num_features):
        self.rng = np.random.default_rng(12345)
        # Small random initialization
        self.beta = self.rng.standard_normal(num_features) * 0.01
        
    def fit(self, X_train, Y_train, lr, tol=1e-4):
        X = np.array(X_train)
        Y = np.array(Y_train)
        n_samples = X.shape[0]
        
        last_loss = float('inf')
        e = float('inf')
        cnt = 0
        
        while e > tol:
            cnt += 1
            
            idx = self.rng.integers(0, n_samples)
            xi = X[idx]
            yi = Y[idx]
            
            # 2. Forward Pass (Single Sample)
            y_hat = np.dot(xi, self.beta)
            
            error = y_hat - yi
            grad = 2 * error * xi
            
            # 4. Update Weights
            self.beta -= lr * grad
            
            if cnt % 5000 == 0:
                # adjust lr
                lr /= 2
                # Calculate current MSE on a larger sample to get a stable reading
                current_loss = np.mean((np.dot(X, self.beta) - Y) ** 2)
                e = abs(last_loss - current_loss)
                last_loss = current_loss
                
                # print(f'Iteration {cnt} - Proxy Loss: {current_loss:.4f} - Change: {e:.6f}')
                
                # Safety break to prevent infinite loops if it diverges
                if np.isnan(current_loss) or cnt > 1000000:
                    print("Stopping: Diverged or reached limit.")
                    break
                    
        return cnt

    def evaluate(self, X_test, Y_test):
        predictions = np.dot(X_test, self.beta)
        return np.mean((predictions - Y_test) ** 2)