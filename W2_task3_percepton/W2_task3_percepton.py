
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.metrics import accuracy_score
import sklearn
import pandas
'''
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = Perceptron()
clf.fit(X, y)
predictions = clf.predict(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''

data_test = pandas.read_csv('perceptron-test.csv', header=None)
data_train = pandas.read_csv('perceptron-train.csv', header=None)
data_test.columns = ["Target", "Feature1", "Feature2" ]
data_train.columns = ["Target", "Feature1", "Feature2" ]
X = data_test[['Feature1', 'Feature2']]
y = data_test['Target']
X_train = data_train[['Feature1', 'Feature2']]
y_train = data_train['Target']
#############before scaling######################
clf = Perceptron()
perc_model = clf.fit(X_train.values, y_train.values)
y_predict = clf.predict(X.values)
score_before = accuracy_score(y, y_predict)
print(score_before)
############after scaling########################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X)
perc_model_scaled = clf.fit(X_train_scaled, y_train.values)
y_predict_scaled = clf.predict(X_scaled)

score_after = accuracy_score(y, y_predict_scaled)
print(score_after)
print((score_after - score_before).round(3))