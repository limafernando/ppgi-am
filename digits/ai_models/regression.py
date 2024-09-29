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