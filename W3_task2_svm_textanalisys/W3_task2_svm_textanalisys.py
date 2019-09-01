from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


answer = []
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
feature_mapping = vectorizer.get_feature_names()
cv_results = pd.DataFrame(gs.cv_results_)
C_best = gs.best_params_.get('C')


clf1 = SVC(kernel='linear', random_state=241, C = 1.0)
svc_model = clf1.fit(X,y)

results = svc_model.coef_[0]
#results = gs.best_estimator_.coef_
row = results.getrow(0).toarray()[0].ravel()
top_ten_indicies = np.argsort(abs(row))[-10:]
top_ten_values = row[top_ten_indicies]

for a in top_ten_indicies:
    print(a, ' ', feature_mapping[a])
    answer = np.append(answer, feature_mapping[a])

answer.sort()
print(answer)