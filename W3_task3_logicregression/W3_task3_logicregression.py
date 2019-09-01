import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

X = pd.read_csv('data-logistic.csv', header=None).loc[:, 1:]
y = pd.read_csv('data-logistic.csv', header=None).loc[:, 0]


def gradient_descent(X_2, y_2, C = 10.0):
    y_predict_list = []
    a=0
    wi = np.array([0.0, 0.0])
    wj = np.array([0.0, 0.0])
    total_rows = X_2.shape[0]
    dst = 1
    k = 0.1
    for iter in range(1000):
        if dst >= 10 ** -5:
            wi = wj
            wj = np.array([0.0, 0.0])
            summ_w1 = 0
            summ_w2 = 0
            for i in range(total_rows):
                summ_w1 += y_2[i] * X_2.values[i][0] * (
                            1 - 1 / (1 + np.exp(-y_2[i] * (wi[0] * X_2.values[i][0] + wi[1] * X_2.values[i][1]))))
                summ_w2 += y_2[i] * X_2.values[i][1] * (
                            1 - 1 / (1 + np.exp(-y_2[i] * (wi[0] * X_2.values[i][0] + wi[1] * X_2.values[i][1]))))
            wj[0] = wi[0] + (k / total_rows) * summ_w1 - k * C * wi[0]
            wj[1] = wi[1] + (k / total_rows) * summ_w2 - k * C * wi[1]
            dst = distance.euclidean(wi, wj)
        else:
            break
    print('Iter = {}'.format(iter))
    for i in range(total_rows):
        a = 1 / (1 + np.exp(-wj[0] * X_2.values[i][0] - wj[1] * X_2.values[i][1]))
        y_predict_list.append(a)
    y_predict = np.array(y_predict_list)
    return y_predict

roc_without_c = roc_auc_score(y, gradient_descent(X, y, C = 0.0))
roc_with_c = roc_auc_score(y, gradient_descent(X, y, C = 10.0))
print(roc_without_c.round(3))
print(roc_with_c.round(3))








