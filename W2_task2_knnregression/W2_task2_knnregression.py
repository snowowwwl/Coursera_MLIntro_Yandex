import pandas
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

score_array = []
data = sklearn.datasets.load_boston()
y = data['target']
X = data['data']
X_scaled = preprocessing.scale(X)
for k in np.linspace(1.0, 10.0, num=200):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p = k)
    boston_model = neigh.fit(X_scaled, y)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    score = cross_val_score(boston_model, X_scaled, y, cv = kf, scoring='neg_mean_squared_error')
    avg_score = float(sum(score))/len(score)
    score_array.append(avg_score)
score_series = pandas.Series(score_array)
print(score_series)
print(np.linspace(1.0, 10.0, num=200))
print(score_series.idxmax(),score_series.max().round(1))