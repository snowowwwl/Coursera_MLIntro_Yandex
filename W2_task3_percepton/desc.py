from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
clf = Perceptron()
X_train = pd.read_csv('perceptron-train.csv', header=None).ix[:, 1:]
X_test = pd.read_csv('perceptron-test.csv', header=None).ix[:, 1:]
y_train = pd.read_csv('perceptron-train.csv', header=None).ix[:, 0]
y_test = pd.read_csv('perceptron-test.csv', header=None).ix[:, 0]
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf.fit(X_train_scaled, y_train)
predict_scaled_data = clf.predict(X_test_scaled)
print('Accuracy after scalibng: ', accuracy_score(y_test, predict_scaled_data))
print('Difference: ', abs(accuracy_score(y_test, predictions) - accuracy_score(y_test, predict_scaled_data)).round(3))