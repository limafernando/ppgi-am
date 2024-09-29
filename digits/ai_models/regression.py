import numpy as np


class LinearRegression:
    def __init__(self) -> None:
        self.w = np.zeros(3)

    def fit(self, _X, _y):
        Xt = _X.T  # transposta de _X
        XtX_inv = np.linalg.inv(Xt @ _X)  # Inversa da transposta de _X por _X
        XtY = Xt @ _y  # transposta de _X por _y
        self.w = XtX_inv @ XtY
     
    def predict(self, _x):
        return np.sign(_x @ self.w)
     
    def getW(self):
        return self.w


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, bs=64):
      self.lr = lr
      self.n_iters = n_iters
      self.batch_size = bs
      self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        epsilon = 1e-9
        
        y_pred = self.sigmoid(np.dot(X, self.w))

        cost = (
            y * np.log(y_pred + epsilon)
            + 
            (1 - y) * np.log(1 - y_pred + epsilon)
        )
        return -np.mean(cost)
    
    # Função para criar minibatches
    def create_minibatches(self, X, y):
        m = X.shape[0]
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, self.batch_size):
            X_batch = X_shuffled[i:i+self.batch_size]
            y_batch = y_shuffled[i:i+self.batch_size]
            yield X_batch, y_batch

    # Infere o vetor w da funçao hipotese
    def fit(self, _X, _y):
        
        X = np.array(_X)
        y = np.array(_y)

        n_features = X.shape[1]

        self.w = np.zeros(n_features)

        for it in range(self.n_iters):
            minibatches = self.create_minibatches(X, y)
            
            for X_batch, y_batch in minibatches:
                n_samples = X_batch.shape[0]
                
                # Calcula a predição para o mini-lote
                A = self.sigmoid(
                    np.dot(X_batch, self.w)
                )

                dz = A - y_batch

                # Computa gradientes
                dw = (1 / n_samples) * np.dot(X_batch.T, dz)
                
                # Atualiza os parâmetros usando o minibatch
                self.w -= self.lr * dw 
            
            if it == 0 or (it+1) % 100 == 0:  # Print cost every 100 iterations
                cost = self.compute_cost(X, y)
                # print(f"Iteration {it+1}, Cost: {cost}")
    
        
    #funcao hipotese inferida pela regressa logistica  
    def predict_prob(self, X):
        return self.sigmoid(
            np.dot(
                np.array(X),
                self.w
            )
        )

    #Predicao por classificação linear
    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
    
    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]
