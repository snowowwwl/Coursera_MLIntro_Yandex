import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

data = pd.read_csv('data-logistic.csv', index_col=False, header=None)

y = data.iloc[:, 0]
X = data.iloc[:, 1:]
w1 = 0  # Вес 1
w2 = 0  # Вес 2
L1 = 0  #
len_y = len(y)


def gradient_spysk(w1, w2, c):
    k = 0.1  # Длинна Шага
    L = 0  #
    i = 0
    while (i < len_y):
        summa = 1 - 1 / (1 + np.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i])))
        w1 = w1 + k / len_y * y[i] * X[1][i] * summa - k * c * w1
        w2 = w2 + k / len_y * y[i] * X[2][i] * summa - k * c * w2
        L = L + (np.log(1 + np.exp(w1 * X[1][i] + w2 * X[2][i])))
        i = i + 1
    return L, w1, w2


def logistic_reqression(c):
    L = 0.0
    w1 = 0.0  # Вес 1
    w2 = 0.0  # Вес 2
    w1_p = 0.0  # Вес old
    w2_p = 0.0  # Вес old
    i = 0
    # Рекомендуется ограничить сверху число итераций десятью тысячами.
    while (i < 10000):
        L, w1, w2 = gradient_spysk(w1, w2, c)
        # евклидово расстояние между векторами
        evk = np.sqrt(pow((w1_p - w1), 2) + pow((w2_p - w2), 2))
        print(i, ' evklid distance: %0.8f' % evk)
        if (evk < 0.00001):
            break
        w1_p = np.copy(w1)
        w2_p = np.copy(w2)
        i = i + 1
    return L, w1, w2


def auc_roc(w1, w2):
    i = 0
    a = []
    while (i < len_y):
        a.append(1 / (1 + np.exp(- w1 * X[1][i] - w2 * X[2][i])))
        i = i + 1
    return roc_auc_score(y, a)


L1, L1_w1, L1_w2 = logistic_reqression(0)
print('Обычная регрессия C=0 L1: ', L1);

L2, L2_w1, L2_w2 = logistic_reqression(10)
print('L2-регуляризованной (с коэффициентом регуляризации 10) L2: ', L2);

L1_rez = auc_roc(L1_w1, L1_w2)
print('AUS-ROC L1: ', L1_rez)

L2_rez = auc_roc(L2_w1, L2_w2)
print('AUS-ROC L2: ', L2_rez)