
import random

import numpy as np

from metrics.metrics import compute_acc

class BasePLA:
    def __init__(self) -> None:
        # w = [0, 0, 0]# w1, w2, theta
        pass

    def constroiListaPCI(self, X, Y, w):
        """
        Esta função constrói a lista de pontos classificados incorretamente.
        
        Paramêtros:
        - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
        às coordenadas dos pontos gerados.
        - Y (list): Classificação dos pontos da amostra X.
        - w (list): Lista correspondendo aos pesos do perceptron.
    
        Retorno:
        - l (list): Lista com os pontos classificador incorretamente.
        - new_y (list): Nova classificação de tais pontos.
    
        """    
        l = []
        new_y = []
        for idx in range(len(X)):
            prediction = np.sign(
                w[0] + w[1]*X[idx][1] + w[2]*X[idx][2]
            )
            # prediction_logit = np.dot(np.asarray(w[0:2]), np.add(X[idx], w[2]))
            # prediction = np.sign(prediction_logit)
            if prediction != Y[idx]:
                l.append(X[idx])
                new_y.append(Y[idx])
        
        return l, new_y


class PLA(BasePLA):  

    def __init__(self) -> None:
        super().__init__()                          

    def fit(self, X, Y, epochs=100):
        """
        Esta função corresponde ao Algoritmo de Aprendizagem do modelo Perceptron.
        
        Paramêtros:
        - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
        às coordenadas dos pontos gerados.
        - y (list): Classificação dos pontos da amostra X.
        - f (list): Lista de dois elementos correspondendo, respectivamente, aos coeficientes angular e linear 
        da função alvo.
        
        Retorno:
        - it (int): Quantidade de iterações necessárias para corrigir todos os pontos classificados incorretamente.
        - w (list): Lista de três elementos correspondendo aos pesos do perceptron.
        """
        
        it = 0
        w = [0, 0, 0]
        W =[w.copy()]
        
        listaPCI, new_y = self.constroiListaPCI(X, Y, w)
        
        while (len(listaPCI) > 0) and it < epochs:
            idx = random.randint(0, len(listaPCI)-1)

            x, y = listaPCI[idx], new_y[idx]

            w[0] = w[0] + y
            w[1] = w[1] + y*x[1] # intensidade
            w[2] = w[2] + y*x[2] # simetria
            
            listaPCI, new_y = self.constroiListaPCI(X, Y, w)
            
            it += 1
            W.append(w.copy())       
            
        return it, w, W     
    
    def predict(self, X, w, single=False):
        
        if single:
            prediction = np.sign(
                w[0] + w[1]*X[1] + w[2]*X[2]
            )
            return prediction
        else:
            Y_pred = []
            for idx in range(len(X)):
                prediction = np.sign(
                    w[0] + w[1]*X[idx][1] + w[2]*X[idx][2]
                )
                Y_pred.append(prediction)
        
            return Y_pred

class PocketPLA(BasePLA):  

    def __init__(self) -> None:
        super().__init__()                           

    def fit(self, X, Y, epochs=100):
        """
        Esta função corresponde ao Algoritmo de Aprendizagem do modelo Perceptron.
        
        Paramêtros:
        - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
        às coordenadas dos pontos gerados.
        - y (list): Classificação dos pontos da amostra X.
        - f (list): Lista de dois elementos correspondendo, respectivamente, aos coeficientes angular e linear 
        da função alvo.
        
        Retorno:
        - it (int): Quantidade de iterações necessárias para corrigir todos os pontos classificados incorretamente.
        - w (list): Lista de três elementos correspondendo aos pesos do perceptron.
        """
        best_w = [0, 0, 0]
        it = 0
        # W =[w.copy()]
        
        listaPCI, new_y = self.constroiListaPCI(X, Y, best_w)

        current_pred = self.predict(X, best_w)
        current_acc = compute_acc(Y, current_pred)
        
        while (len(listaPCI) > 0) and it < epochs:
            new_w = [None, None, None]

            idx = random.randint(0, len(listaPCI)-1)

            x, y = listaPCI[idx], new_y[idx]

            new_w[0] = best_w[0] + y
            new_w[1] = best_w[1] + y*x[1] # intensidade
            new_w[2] = best_w[2] + y*x[2] # simetria

            new_pred = self.predict(X, new_w)
            new_acc = compute_acc(Y, new_pred)

            if new_acc > current_acc:
                current_acc = new_acc
                current_pred = new_pred
                best_w = new_w.copy()
            
            listaPCI, new_y = self.constroiListaPCI(X, Y, best_w)
            
            it += 1
            # W.append(w.copy())       
            
        return it, best_w#, W       
    
    def predict(self, X, w):
        Y_pred = []
        for idx in range(len(X)):
            prediction = np.sign(
                w[1]*X[idx][1] + w[2]*X[idx][2] + w[0] # x2-(m*x1)-b
            )
            Y_pred.append(prediction)
        return Y_pred
    