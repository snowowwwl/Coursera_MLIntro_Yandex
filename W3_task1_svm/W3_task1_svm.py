from sklearn.svm import SVC
import pandas as pd


X = pd.read_csv('svm-data.csv', header=None).loc[:, 1:]
y = pd.read_csv('svm-data.csv', header=None).loc[:, 0]
clf = SVC(C = 100000, random_state=241, kernel='linear')
clf.fit(X, y)
print(X)
print(clf.support_)