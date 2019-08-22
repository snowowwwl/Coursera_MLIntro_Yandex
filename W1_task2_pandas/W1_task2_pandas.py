import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print("************Q1:")
'''
Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
'''
sex = data.groupby(['Sex', 'Survived']).count()
print(sex)
sex = data.groupby('Sex').count()
print(sex)

print("*************Q2:")
'''
Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. Ответ приведите в процентах 
(число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
'''
survived = data.groupby('Survived').count()
print(data['Survived'].count())
print(survived)
print((survived['Pclass']/data['Survived'].count()).round(2))

print("*************Q3:")
'''
Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах 
(число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
'''
pclass = data.groupby('Pclass').count()
print(data['Pclass'].count())
print(pclass)
print((pclass['Survived']/data['Pclass'].count()).round(4)*100)

print("*************Q4:")
'''
Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. 
В качестве ответа приведите два числа через пробел.
'''

print(data['Age'].mean().round(2))
print(data['Age'].median())

print("*************Q5:")
'''
Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? Посчитайте корреляцию 
Пирсона между признаками SibSp и Parch.
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
'''
print((data['SibSp'].corr(data['Parch'])).round(2))

print("*************Q6:")
'''
Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя 
(First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных. 
Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. Попробуйте вручную разобрать 
несколько значений столбца 
Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.'''

#print(data.Name.mode.value_counts())

name_list = []
gname_list = []
import re
for n in data.Name:
    name = re.search('\((\S+) ', n)
    if name:
        name_list.append(name.group(1))
for n in data.Name:
    g_name = re.search('Miss. (\S+) ', n)
    if g_name:
        gname_list.append(g_name.group(1))
name_list = name_list + gname_list
data_name = pandas.Series(name_list)
print(data_name)
print(data_name.mode())

