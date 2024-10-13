from matplotlib import pyplot as plt
import pandas as pd


COLOR_MAP = {
    0: "blue",
    1: "green",
    4: "red",
    5: "gold",
}

MARKERS_MAP = {
    0: "o",
    1: "P",
    4: "s",
    5: "p",
}


def show_bin_dataset(X: pd.DataFrame, Y: pd.DataFrame, d1: int, d2: int, columns: list):

    c1, m1 = COLOR_MAP[d1], MARKERS_MAP[d1]
    c2, m2 = COLOR_MAP[d2], MARKERS_MAP[d2]

    df = pd.concat([X, Y], axis=1)

    df1 = df[(df["label"] == d1)]
    df2 =  df[(df["label"] == d2)]
    
    df = pd.concat([df1, df2]).reset_index()

    for i in range(len(df)):
        if (df["label"][i] == d1):
            plt.scatter(df[columns[0]][i], df[columns[1]][i], c=c1, marker=m1)
        else:
            plt.scatter(df[columns[0]][i], df[columns[1]][i], c=c2, marker=m2)

    scatter1 = plt.Line2D([0], [0], marker=m1, color='w', markerfacecolor=c1, markersize=10, label=f'Class {d1}')
    scatter2 = plt.Line2D([0], [0], marker=m2, color='w', markerfacecolor=c2, markersize=10, label=f'Class {d2}')

    plt.legend(handles=[scatter1, scatter2], loc='upper left')

    plt.xlabel(columns[0])
    plt.ylabel(columns[1])

    plt.show()


def show_dataset(X: pd.DataFrame, Y: pd.DataFrame, columns: list):

    c0, m0 = COLOR_MAP[0], MARKERS_MAP[0]
    c1, m1 = COLOR_MAP[1], MARKERS_MAP[1]
    c4, m4 = COLOR_MAP[4], MARKERS_MAP[4]
    c5, m5 = COLOR_MAP[5], MARKERS_MAP[5]


    df = pd.concat([X, Y], axis=1)

    for i in range(len(df)):
        if (df["label"][i] == 0):
            plt.scatter(df[columns[0]][i], df[columns[1]][i], c=c0, marker=m0)
        if (df["label"][i] == 1):
            plt.scatter(df[columns[0]][i], df[columns[1]][i], c=c1, marker=m1)
        if (df["label"][i] == 4):
            plt.scatter(df[columns[0]][i], df[columns[1]][i], c=c4, marker=m4)
        if (df["label"][i] == 5):
            plt.scatter(df[columns[0]][i], df[columns[1]][i], c=c5, marker=m5)

    scatter0 = plt.Line2D([0], [0], marker=m0, color='w', markerfacecolor=c0, markersize=10, label=f'Class {0}')
    scatter1 = plt.Line2D([0], [0], marker=m1, color='w', markerfacecolor=c1, markersize=10, label=f'Class {1}')
    scatter4 = plt.Line2D([0], [0], marker=m4, color='w', markerfacecolor=c4, markersize=10, label=f'Class {4}')
    scatter5 = plt.Line2D([0], [0], marker=m5, color='w', markerfacecolor=c5, markersize=10, label=f'Class {5}')

    plt.legend(handles=[scatter0, scatter1, scatter4, scatter5], loc='upper left')

    plt.xlabel(columns[0])
    plt.ylabel(columns[1])

    plt.show()
