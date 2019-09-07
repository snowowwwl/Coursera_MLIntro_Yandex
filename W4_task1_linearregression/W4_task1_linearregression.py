import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import  hstack
from sklearn.linear_model import Ridge

data_raw = pd.read_csv('salary-train.csv')
data_predict = pd.read_csv('salary-test-mini.csv')

# preprocessing
# to lower case
data_raw['FullDescription'] = data_raw['FullDescription'].str.lower()
data_raw['LocationNormalized'] = data_raw['LocationNormalized'].str.lower()
data_raw['ContractTime'] = data_raw['ContractTime'].str.lower()

data_predict['FullDescription'] = data_predict['FullDescription'].str.lower()
data_predict['LocationNormalized'] = data_predict['LocationNormalized'].str.lower()
data_predict['ContractTime'] = data_predict['ContractTime'].str.lower()

# to letters and numbers
data_raw['FullDescription'] = data_raw['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_raw['LocationNormalized'] = data_raw['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_raw['ContractTime'] = data_raw['ContractTime'].replace('[^a-zA-Z0-9]', ' ', regex = True)

data_predict['FullDescription'] = data_predict['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_predict['LocationNormalized'] = data_predict['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_predict['ContractTime'] = data_predict['ContractTime'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# vectorize text
vectorizer = TfidfVectorizer(min_df=5)
descr = vectorizer.fit_transform(data_raw['FullDescription'])
descr_predict = vectorizer.transform(data_predict['FullDescription'])

# empty to nan
data_raw['LocationNormalized'].fillna('nan', inplace=True)
data_raw['ContractTime'].fillna('nan', inplace=True)

data_predict['LocationNormalized'].fillna('nan', inplace=True)
data_predict['ContractTime'].fillna('nan', inplace=True)

#todict
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_raw[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_predict[['LocationNormalized', 'ContractTime']].to_dict('records'))


#concatenate
X = hstack([descr, X_train_categ])
X_predict = hstack([descr_predict, X_test_categ])
y = data_raw['SalaryNormalized']



#fit
clf = Ridge(alpha=1.0, random_state=241)
ridgemodel = clf.fit(X, y.values)


#predict
y_predict = ridgemodel.predict(X_predict)
print(y_predict.round(2))