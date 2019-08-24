
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas


score_array = []
score_array_scaled = []
data = pandas.read_csv('wine.data', sep = ',')
data.columns = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                "OD280/OD315 of diluted wines", "Proline" ]
X = data[["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
          "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
          "OD280/OD315 of diluted wines", "Proline"]]
y = data['Class']
##############before scale####################
for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    wine_model = neigh.fit(X.values, y.values)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    score = cross_val_score(wine_model, X, y, cv = kf)
    avg_score = float(sum(score))/len(score)
    score_array.append(avg_score)
score_series = pandas.Series(score_array)
print(score_series.idxmax(),score_series.max().round(2) )

##############after scale####################
X_scaled = preprocessing.scale(X)
for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    wine_model = neigh.fit(X_scaled, y.values)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    score = cross_val_score(wine_model, X_scaled, y, cv = kf)
    avg_score = float(sum(score))/len(score)
    score_array_scaled.append(avg_score)
score_series_scaled = pandas.Series(score_array_scaled)
print(score_series_scaled.idxmax(),score_series_scaled.max().round(2) )
