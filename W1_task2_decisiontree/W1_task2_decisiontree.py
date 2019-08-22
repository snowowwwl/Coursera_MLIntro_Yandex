'''
Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
Обратите внимание, что признак Sex имеет строковые значения.
Выделите целевую переменную — она записана в столбце Survived.
В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении
их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах
конструктора DecisionTreeСlassifier).
Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи
 (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).
'''
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydot
import pydotplus
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from io import StringIO


transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data_sample = data[['Sex', 'Pclass', 'Age', 'Fare', 'Survived']]
data_sample = data_sample.dropna()
data_sample1 = data_sample[['Sex', 'Pclass', 'Age', 'Fare']]
survived_target = data_sample['Survived']

x = transformer.fit_transform(data_sample1)
print(data_sample1)
print(x)


survived_model = DecisionTreeClassifier(random_state=241)
survived_model.fit(x, survived_target.values)
importances = survived_model.feature_importances_
print(importances)
dot_data = StringIO()
out = tree.export_graphviz(survived_model, feature_names=['Sex', 'Sex', 'Pclass', 'Age', 'Fare'],
out_file = dot_data, filled= True )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('survived_model.pdf')