import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def compute_acc(Y, Y_pred):
    acc = 0
    sample_size = len(Y)
    for idx in range(sample_size):
        if Y_pred[idx] == Y[idx]:
            acc += 1
    return acc/sample_size

def confusion(Y, Y_pred):
    cm = confusion_matrix(Y, Y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 1], yticklabels=[-1, 1])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def eficiency_report(Y, Y_pred):
    return {
        "acc": accuracy_score(Y, Y_pred),
        "precision": precision_score(Y, Y_pred),
        "recall": recall_score(Y, Y_pred),
        "f1": f1_score(Y, Y_pred),
    }
