import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
score_list=[]

kf = KFold(n_splits=5, random_state=1, shuffle=True)

for k in range(1,51):
    clf = RandomForestRegressor(n_estimators=k, random_state=1)
    forest_model = clf.fit(X, y)
    """
    Ne sovpalo s otvetom:
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        score = r2_score(y_predict,y_test)
        score_list.append(score)
    avg_score = sum(score_list) / 5
    """
    score = cross_val_score(forest_model, X, y, cv=kf, scoring='r2')
    avg_score = (sum(score)) / len(score)
    score_list.append(avg_score.round(2))
print(score_list)
score_series = pd.Series(score_list)
print(score_series.where(score_series > 0.52))