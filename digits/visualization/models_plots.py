from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


COLOR_MAP = {
    0: "blue",
    1: "green",
    5: "gold",
    4: "red",
}

MARKERS_MAP = {
    0: "o",
    1: "P",
    5: "p",
    4: "s",
}

def show_model_line(X: pd.DataFrame, Y: pd.DataFrame, d1: int, d2: int, w):

    c1, m1 = COLOR_MAP[d1], MARKERS_MAP[d1]
    c2, m2 = COLOR_MAP[d2], MARKERS_MAP[d2]

    # Y = Y.replace(1, d1)
    # Y = Y.replace(-1, d2)

    max_x1 = X["intensity"].max()
    max_x2 = X["symmetry"].max()

    df = pd.concat([X, Y], axis=1)

    # df1 = df[(df["label"] == d1)]
    # df2 =  df[(df["label"] == d2)]
    
    # df = pd.concat([df1, df2]).reset_index()

    # plt.axis([-1, 1, -1, 1])

    for i in range(len(df)):
        if (df["label"][i] == d1):
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=c1, marker=m1)
        else:
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=c2, marker=m2)

    scatter1 = plt.Line2D([0], [0], marker=m1, color='w', markerfacecolor=c1, markersize=10, label=f'Class {d1}')
    scatter2 = plt.Line2D([0], [0], marker=m2, color='w', markerfacecolor=c2, markersize=10, label=f'Class {d2}')

    x = np.linspace(0, 120, 1000)
    plt.plot(x, (-w[0] - w[1]*x) / w[2], c='orange') # w = [0, 0, 0] # w1, w2, theta  

    plt.legend(handles=[scatter1, scatter2], loc='upper left')

    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')

    plt.xlim(0, max_x1+50)
    plt.ylim(0, max_x2+50)


    plt.show()


def plot_grafico(X, y, w, f):   
    """
    Esta função objetiva a visualização dos passos do PLA.
    
    Paramêtros:
    - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
    às coordenadas dos pontos gerados.
    - y (list): Classificação dos pontos da amostra X.
    - w (list): Lista correspondendo aos pesos do perceptron.
    - f (list): Lista de dois elementos correspondendo, respectivamente, aos coeficientes angular e linear 
    da função alvo.    
    """
    
    nPontos = len(X)    
    #matplotlib.use('TkAgg')    
            
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')

    plt.axis([-1, 1, -1, 1])

    x_plt = [X[i][0] for i in range(nPontos)]
    y_plt = [X[i][1] for i in range(nPontos)]

    for i in range(nPontos):
        if (y[i] == 1):
            plt.scatter(x_plt[i], y_plt[i], s=10, c='blue')
        else:
            plt.scatter(x_plt[i], y_plt[i], s=10, c='red')

    x = np.linspace(-1, 1, 1000)
    plt.plot(x, f[0]*x + f[1], c='green') # f[0] = m, f[1] = b
    # plt.plot(x, (-w[0] - w[1]*x) / w[2], c='orange') # A*x + B*y + C => y = (-C - A*x) / B
    plt.plot(x, (-w[2] - w[0]*x) / w[1], c='orange') # w = [0, 0, 0] # w1, w2, theta
    # clear_output(wait=True)    
    plt.show(block=False)    
    plt.pause(0.01)   