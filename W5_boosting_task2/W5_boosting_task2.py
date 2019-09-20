import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('gbm-data.csv')
y = data.values[:,0]
X = data.values[:,1:]
learning_rate = [0.2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
a_train_list=[]
a_test_list = []
logloss_train=[]
logloss_test = []
for lr in learning_rate:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate = lr)
    boosting = clf.fit(X_train, y_train)
    for i, y_pred_train in enumerate(clf.staged_decision_function(X_train)) :
        a_train_list.clear()
        for y_pred in y_pred_train:
            a_train = 1 / (1 + math.exp(-(y_pred)))
            a_train_list.append(a_train)
        a_train_array = np.asarray(a_train_list)
        logloss_train.append(log_loss(y_train, a_train_array))

    for i, y_pred_test in enumerate(clf.staged_decision_function(X_test)):
        a_test_list.clear()
        for y_pred in y_pred_test:
            a_test = 1 / (1 + math.exp(-(y_pred)))
            a_test_list.append(a_test)
        a_test_array = np.array(a_test_list)
        logloss_test.append(log_loss(y_test, a_test_array))
    ll_train = pd.Series(logloss_train)
    print(ll_train.idxmin(), ll_train.min().round(2))
    ll_test = pd.Series(logloss_test)
    print(ll_test.idxmin(), ll_test.min().round(2))
    plt.figure()
    plt.plot(range(len(logloss_test)),logloss_test, 'r', linewidth=2)
    plt.plot(range(len(logloss_train)),logloss_train, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

######compare with forest#####
clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)
pred_train = clf.predict_proba(X_train)
pred_test = clf.predict_proba(X_test)
print(log_loss(y_test, pred_test).round(2))


