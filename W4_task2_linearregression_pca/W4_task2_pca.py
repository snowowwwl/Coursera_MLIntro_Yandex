import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

X = pd.read_csv('close_prices.csv').iloc[:, 1:]
y = pd.read_csv('djia_index.csv').iloc[:, 1:]
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)

Xtr = pca.transform(X)


corr = np.corrcoef(Xtr[:,0], y.values[:,0])
print(corr.round(2))

a = np.array(pca.components_[0])
print(a.max(), a.argmax())
print(X.columns[26])
