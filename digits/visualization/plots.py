from random import random

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


def show_bin_dataset(X, Y, d1, d2):
    """
    Esta função tem o objetivo de exibir na tela uma amostra do dataset passado por parâmetro.
    
    Paramêtros:
    - X (matriz): Matriz 1000x2 correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
    às coordenadas dos pontos gerados.
    - y (list): Classificação dos pontos da amostra X.
    - p1 (list): Coordenadas do ponto p1 gerado para criar a função alvo.
    - p2 (list): Coordenadas do ponto p2 gerado para criar a função alvo.
    - m (float): Coeficiente angular da função alvo.
    - b (float): Coeficidnte linear da função alvo.
    """
    
    # line = np.linspace(-1, 1, 1000) 
    # plt.plot(line, m*line + b, label="f(x)", c="green")

    # Pontos usados na criacao da reta
    # plt.scatter(p1[0], p1[1], c='green')
    # plt.scatter(p2[0], p2[1], c='green')

    c1, m1 = COLOR_MAP[d1], MARKERS_MAP[d1]
    c2, m2 = COLOR_MAP[d2], MARKERS_MAP[d2]

    df = pd.concat([X, Y], axis=1)

    df1 = df[(df["label"] == d1)]
    df2 =  df[(df["label"] == d2)]
    
    df = pd.concat([df1, df2]).reset_index()

    for i in range(len(df)):
        if (df["label"][i] == d1):
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=c1, marker=m1)
        else:
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=c2, marker=m2)

    scatter1 = plt.Line2D([0], [0], marker=m1, color='w', markerfacecolor=c1, markersize=10, label=f'Class {d1}')
    scatter2 = plt.Line2D([0], [0], marker=m2, color='w', markerfacecolor=c2, markersize=10, label=f'Class {d2}')

    plt.legend(handles=[scatter1, scatter2], loc='upper left')

    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')

    plt.show()
