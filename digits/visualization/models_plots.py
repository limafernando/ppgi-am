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

def show_model_line(X: pd.DataFrame, Y: pd.DataFrame, w: list, d1: int, d2: int, d3: int, d4: int):

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
        if (d1 is not None and df["label"][i] == d1):
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=c1, marker=m1)
        if (d2 is not None and df["label"][i] == d2):
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=c2, marker=m2)
        if (d3 is not None and df["label"][i] == d3):
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=COLOR_MAP[d3], marker=MARKERS_MAP[d3])
        if (d4 is not None and df["label"][i] == d4):
            plt.scatter(df["intensity"][i], df["symmetry"][i], c=COLOR_MAP[d4], marker=MARKERS_MAP[d4])

    valid_digs = [d1, d2, d3, d4]
    vds = len([x for x in valid_digs if x is not None])

    scatters = []
    for i in range(vds):
        scatter = plt.Line2D([0], [0], marker=MARKERS_MAP[valid_digs[i]], color='w', markerfacecolor=COLOR_MAP[valid_digs[i]], markersize=10, label=f'Class {valid_digs[i]}')
        scatters.append(scatter)


    # scatter1 = plt.Line2D([0], [0], marker=m1, color='w', markerfacecolor=c1, markersize=10, label=f'Class {d1}')
    # scatter2 = plt.Line2D([0], [0], marker=m2, color='w', markerfacecolor=c2, markersize=10, label=f'Class {d2}')

    x = np.linspace(0, 250, 1000)
    plt.plot(x, (-w[0] - w[1]*x) / w[2], c='orange') # w = [0, 0, 0] # w1, w2, theta  

    plt.legend(handles=scatters, loc='upper left')

    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')

    plt.xlim(0, max_x1+50)
    plt.ylim(0, max_x2+50)


    plt.show()


def encontrar_intersecao(m1, b1, m2, b2):
    x_intersecao = (b2 - b1) / (m1 - m2)
    y_intersecao = m1 * x_intersecao + b1
    return x_intersecao, y_intersecao

def show_mult_model_lines(X: pd.DataFrame, Y: pd.DataFrame, columns: list, Ws: list, Ds:list = [1,0,4,5]):

    c0, m0 = COLOR_MAP[0], MARKERS_MAP[0]
    c1, m1 = COLOR_MAP[1], MARKERS_MAP[1]
    c4, m4 = COLOR_MAP[4], MARKERS_MAP[4]
    c5, m5 = COLOR_MAP[5], MARKERS_MAP[5]

    max_x1 = X["intensity"].max()
    max_x2 = X["symmetry"].max()

    df = pd.concat([X, Y], axis=1).reset_index()

    fig, ax = plt.subplots()

    for i in range(len(df)):
        if (df["label"][i] == 0):
            plt.scatter(df[columns[1]][i], df[columns[2]][i], c=c0, marker=m0)
        if (df["label"][i] == 1):
            plt.scatter(df[columns[1]][i], df[columns[2]][i], c=c1, marker=m1)
        if (df["label"][i] == 4):
            plt.scatter(df[columns[1]][i], df[columns[2]][i], c=c4, marker=m4)
        if (df["label"][i] == 5):
            plt.scatter(df[columns[1]][i], df[columns[2]][i], c=c5, marker=m5)

    scatter0 = plt.Line2D([0], [0], marker=m0, color='w', markerfacecolor=c0, markersize=10, label=f'Class {0}')
    scatter1 = plt.Line2D([0], [0], marker=m1, color='w', markerfacecolor=c1, markersize=10, label=f'Class {1}')
    scatter4 = plt.Line2D([0], [0], marker=m4, color='w', markerfacecolor=c4, markersize=10, label=f'Class {4}')
    scatter5 = plt.Line2D([0], [0], marker=m5, color='w', markerfacecolor=c5, markersize=10, label=f'Class {5}')

    x = np.linspace(0, 400, 1000)
    cc = 0
    retas = []
    for w in Ws:
        r = (-w[0] - w[1]*x) / w[2]
        retas.append(r)
        plt.plot(
            x,
            r,
            c=COLOR_MAP[Ds[cc]]
        ) # w = [0, 0, 0] # w1, w2, theta
        cc += 1
        

    # interseccoes = []

    # x12, y12 = encontrar_intersecao(Ws[0][1], Ws[0][2], Ws[1][1], Ws[1][2])
    # x23, y23 = encontrar_intersecao(Ws[1][1], Ws[1][2], Ws[2][1], Ws[2][2])

    # x1 = np.linspace(0, x12, 1000)   # Linha 1 até a interseção com Linha 2
    # x2 = np.linspace(x12, x23, 1000) # Linha 2 entre as interseções com Linha 1 e Linha 3
    # x3 = np.linspace(x23, 10, 1000)  # Linha 3 a partir da interseção com Linha 2
    # Xs = [x1,x2,x3]

    # for i in range(len(Ws)):
    #     r = (-Ws[i][0] - Ws[i][1]*Xs[i]) / Ws[i][2]
    #     retas.append(r)
    #     plt.plot(
    #         Xs[i],
    #         r,
    #         c=COLOR_MAP[Ds[cc]]
    #     ) # w = [0, 0, 0] # w1, w2, theta 
    #     cc += 1
    
    
    # ax.fill_between(x, retas[0], max(retas[0])+1, color=COLOR_MAP[1], alpha=0.3)
    # ax.fill_between(x, retas[1], retas[0], color=COLOR_MAP[0], alpha=0.3)
    # ax.fill_between(x, retas[2], retas[1], color=COLOR_MAP[4], alpha=0.3)

    
    plt.legend(handles=[scatter0, scatter1, scatter4, scatter5], loc='upper left')

    plt.xlabel(columns[1])
    plt.ylabel(columns[2])

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