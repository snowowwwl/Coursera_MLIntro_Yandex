
import time
import datetime
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

######Gradient boosting##############
"""
1.Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. Удалите признаки, 
связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
"""
features = pd.read_csv('features.csv', index_col='match_id')
features.head()
data = features.iloc[:,:-6]

"""
2. Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число 
заполненных значений. Много ли пропусков в данных? Запишите
названия признаков, имеющих пропуски, и попробуйте для любых двух из них дать обоснование, почему их значения могут
быть пропущены.
"""
print("*********************GRADIENT_BOOSTING_TASK2***************")
for i in range(data.shape[1]):
    if (data.iloc[:,i].count()<data.shape[0]):
        print(data.columns[i], data.shape[0]-data.iloc[:,i].count())
"""
Ответ на вопрос2:
first_blood_time 19553
first_blood_team 19553
first_blood_player1 19553
first_blood_player2 43987
radiant_bottle_time 15691
radiant_courier_time 692
radiant_flying_courier_time 27479
radiant_first_ward_time 1836
dire_bottle_time 16143
dire_courier_time 676
dire_flying_courier_time 26098
dire_first_ward_time 1826

first_blood_time - 19553 пропусков. За первые пять минут матча никого не убили.
radiant_courier_time 692 пропусков - в этих матчах команда radiant не покупала courier в первые 5 минут матча
"""

"""
3.Замените пропуски на нули с помощью функции fillna().
"""
X = data.fillna(0)

"""
4. Какой столбец содержит целевую переменную? Запишите его название.
Ответ на вопрос 4: radiant_win
"""
y = features['radiant_win']

"""
5. Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на имеющейся
 матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не забудьте 
 перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени, и без перемешивания 
 можно столкнуться с нежелательными эффектами при оценивании качества. Оцените качество градиентного бустинга 
 (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при этом разное количество деревьев 
 (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). 
 Долго ли настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же 
 качество, скорее всего, продолжит расти при дальнейшем его увеличении?
"""
print("*********************GRADIENT_BOOSTING_TASK5***************")
score_list = []
roc_auc_list = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
n_list = [30]
for n in n_list:
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators=n, random_state=1, learning_rate = 0.2, max_depth=3)
    gb_model = clf.fit(X, y)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    ####считаю roc-auc двумя способами - cross_val_score
    score = cross_val_score(gb_model, X, y, cv=kf, scoring='roc_auc')
    avg_score = (sum(score)) / len(score)
    #### и roc_auc_score
    pred = clf.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, pred)
    roc_auc_list.append(roc_auc.round(3))
    score_list.append(avg_score.round(3))
    print("The quantity of trees is {}. Metric is:".format(n))
    print(score_list, roc_auc_list)
"""
Ответ на вопрос 5:
10 - Time elapsed: 0:00:07.902452
20 - Time elapsed: 0:00:16.191926
30 - Time elapsed: 0:00:23.140323
metric cross_val_score:
10 - 0.678
20 - 0.692
30 - 0.698
metric roc_auc:
10 - 0.685
20 - 0.702
30 -0.711

max-deph = 3, n = 100:
0:01:28.714074
cross_val_score = 0.712 
roc_auc = 0.736

max-deph = 2, n = 100:
0:00:43.160468
cross_val_score = 0.707 
roc_auc = 0.720

max-deph = 2, n = 150
Time elapsed: 0:01:01.261504
cross_val_score = 0.712
roc_auc = 0.727

При росте n растет качество, но сильнее растет время работы. Можно уменьшить время засчет уменьшения глубины деревьев с 3 до 2х,
но тогда упадет и качество. Думаю, можно оставить n равныт от 30 до 50.
"""

##############Logistic Regression
"""
1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с помощью 
кросс-валидации по той же схеме, которая использовалась для градиентного бустинга. Подберите при этом лучший параметр 
регуляризации (C). Какое наилучшее качество у вас получилось? Как оно соотносится с качеством градиентного бустинга? 
Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
"""
print("*********************LOGISTIC_REGRESSION_TASK1***************")
score_lg_list = []
roc_auc_lg_list = []
kflg = KFold(n_splits=5, random_state=1, shuffle=True)
scaler = StandardScaler()
scaler.fit(X)
Xs = scaler.transform(X)

#n_lg_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
n_lg_list = [0.001]
for n in n_lg_list:
    start_time = datetime.datetime.now()
    clf = LogisticRegression(random_state=1, C = n, solver = "lbfgs", penalty = "l2")
    logr_model = clf.fit(Xs,y)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    ####считаю roc-auc двумя способами - cross_val_score
    score = cross_val_score(logr_model, Xs, y, cv=kflg, scoring='roc_auc')
    avg_score = (sum(score)) / len(score)
    score_lg_list.append(avg_score.round(3))
    #### и roc_auc_score
    pred = clf.predict_proba(Xs)[:, 1]
    roc_auc = roc_auc_score(y, pred)
    roc_auc_lg_list.append(roc_auc.round(3))
    print("C is {}. Metric is:".format(n))
    print(avg_score.round(3), roc_auc.round(3))
"""
Ответ на вопрос 1:
Лучшее значение метрики cross_val_score = 0.716, roc_auc = 0.718 при С = 0.001
Время 0:00:01.080061
Значение метрики при градиентном бустинге n = 30 меньше:
30 - Time elapsed: 0:00:23.140323
0.698 или 0.711
Близкого качества удается добиться в бустинге при:
max-deph = 3, n = 100:
0:01:28.714074 - но при этом время 1м 29 секунд.
[0.712] [0.736]
Логистическая регрессия отработала быстрее и качество получилось выше. Качество,возможно, объясняется регуляризацией.
Быстрота, возможно, объясняется тем, что в памяти алгоритму не нужно держать всю выборку, он может считывать по одному объекты.
"""

"""
2.Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является хорошей идеей.
Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ...,
d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором 
лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?

Ответ на вопрос2: Не улучшилось - видимо,удаление категориальных признаков не так эффективно,как кодирование. 
Потому что это важные признаки.
"""

print("*********************LOGISTIC_REGRESSION_TASK2***************")

data2 = data.drop(['lobby_type','r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], 1)
X = data2.fillna(0)
scaler.fit(X)
Xs = scaler.transform(X)
n = 0.001
start_time = datetime.datetime.now()
clf = LogisticRegression(random_state=1, C = 0.001, solver = "lbfgs", penalty = "l2")
logr_model = clf.fit(Xs,y)
print('Time elapsed:', datetime.datetime.now() - start_time)
####считаю roc-auc двумя способами - cross_val_score
score = cross_val_score(logr_model, Xs, y, cv=kflg, scoring='roc_auc')
avg_score = (sum(score)) / len(score)
#### и roc_auc_score
pred = clf.predict_proba(Xs)[:, 1]
roc_auc = roc_auc_score(y, pred)
print("C is {}. Metric is:".format(n))
print(avg_score.round(3), roc_auc.round(3))
"""
3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои играли 
за каждую команду. Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, 
чем другие. Выясните из данных, сколько различных идентификаторов героев существует в данной игре 
вам может пригодиться фукнция unique или value_counts).

Ответ на вопрос 3: 108 в обучающей выборке, всего в файле heroes = 112

"""

print("*********************LOGISTIC_REGRESSION_TASK3***************")

catgrcl_list = ['r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
unique_list = []
for header in catgrcl_list:
    unique_list.extend(data[header].unique())
unique_ser = pd.Series(unique_list)
unique_ser = unique_ser.unique()
print("Unique id of heroes in train data {}".format(len(unique_ser)))
heroes = pd.read_csv('heroes.csv')
print("Unique id of heroes in dictionaries/heroes {}".format(heroes["id"].count()))
N = heroes["id"].count()


"""
4. Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных героев.
 Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице, 
 если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire. 
 Ниже вы можете найти код, который выполняет данной преобразование. Добавьте полученные признаки к числовым, 
 которые вы использовали во втором пункте данного этапа.

"""


X_pick = np.zeros((data.shape[0], N))

for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_X_pick = np.concatenate((X.values, X_pick), axis=1)


"""
5. Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации. 
Какое получилось качество? Улучшилось ли оно? Чем вы можете это объяснить?

Ответ на вопрос 5: Качество улучшилось - по кроссвалидации 0.752, по roc-auc 0.754 при С=0.001. Улучшение качества можно объяснить
кодированием категориальных признаков, потому что логистическая регрессия чувствиетльна к формату данных и к масштабированию.

"""
print("*********************LOGISTIC_REGRESSION_TASK5***************")
score_lg_list = []
roc_auc_lg_list = []
kflg = KFold(n_splits=5, random_state=1, shuffle=True)
scaler = StandardScaler()
scaler.fit(X_X_pick)
X_X_s = scaler.transform(X_X_pick)

n_lg_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
for n in n_lg_list:
    start_time = datetime.datetime.now()
    clf = LogisticRegression(random_state=1, C = n, solver = "lbfgs", penalty = "l2")
    logr_model = clf.fit(X_X_s,y)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    ####считаю roc-auc двумя способами - cross_val_score
    score = cross_val_score(logr_model, X_X_s, y, cv=kflg, scoring='roc_auc')
    avg_score = (sum(score)) / len(score)
    score_lg_list.append(avg_score.round(3))
    #### и roc_auc_score
    pred = clf.predict_proba(X_X_s)[:, 1]
    roc_auc = roc_auc_score(y, pred)
    roc_auc_lg_list.append(roc_auc.round(3))
    print("****C is {}. Metric is:".format(n))
    print(avg_score.round(3), roc_auc.round(3))

"""
6. Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных моделей 
(лучшей с точки зрения AUC-ROC на кросс-валидации). Убедитесь, что предсказанные вероятности адекватные — находятся на 
отрезке [0, 1], не совпадают между собой (т.е. что модель не получилась константной).
Ответ на задание 6:
Max proba 0.99 in match id 33469
Min proba 0.005 in match id 14176
"""
print("*********************LOGISTIC_REGRESSION_TASK6***************")
X_test = pd.read_csv('features_test.csv', index_col='match_id')
X_6 = X_test.drop(['lobby_type','r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], 1)
X_6 = X_6.fillna(0)

X_pick = np.zeros((X_test.shape[0], N))
for i, match_id in enumerate(X_test.index):
    for p in range(5):
        X_pick[i, X_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X6_X_pick = np.concatenate((X_6.values, X_pick), axis=1)
kflg = KFold(n_splits=5, random_state=1, shuffle=True)
scaler = StandardScaler()
scaler.fit(X6_X_pick)
X6_X_s = scaler.transform(X6_X_pick)

start_time = datetime.datetime.now()
clf = LogisticRegression(random_state=1, C = 0.001, solver = "lbfgs", penalty = "l2")
logr_model = clf.fit(X_X_s,y)
print('Time elapsed:', datetime.datetime.now() - start_time)
y_test = clf.predict_proba(X6_X_s)
X_test['y_test'] = y_test[:,0]
print("Max proba {} in match id {}".format(X_test["y_test"].max().round(3),X_test["y_test"].idxmax()))
print("Min proba {} in match id {}".format(X_test["y_test"].min().round(3),X_test["y_test"].idxmin()))