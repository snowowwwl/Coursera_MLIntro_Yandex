import pandas as pd
import numpy as np
from  sklearn.metrics import accuracy_score
from  sklearn.metrics import precision_score
from  sklearn.metrics import recall_score
from  sklearn.metrics import f1_score
from  sklearn.metrics import roc_auc_score
from  sklearn.metrics import precision_recall_curve

y_true = pd.read_csv('classification.csv')['true']
y_pred = pd.read_csv('classification.csv')['pred']

##############task1###############

c =(y_true*y_pred)
TP = c.sum()
c = (y_true>y_pred)
FN = c.sum()
c = (y_true<y_pred)
FP = c.sum()
c = np.logical_or(y_true,y_pred)
d = np.logical_not(c)
TN = d.sum()
print('TP = {}, FP = {} FN = {} TN = {} SUMM = {}'.format(TP, FP, FN, TN, TP+FN+FP+TN))

##############task2###############
ACC_SCORE = accuracy_score(y_true, y_pred)
print("ACC_SCORE = {}".format(ACC_SCORE.round(2)))

PRE_SCORE = precision_score(y_true, y_pred)
print("PRE_SCORE = {}".format(PRE_SCORE.round(2)))

REC_SCORE = recall_score(y_true, y_pred)
print("REC_SCORE = {}".format(REC_SCORE.round(2)))

F1_SCORE = f1_score(y_true, y_pred)
print("F1_SCORE = {}".format(F1_SCORE.round(2)))

############task3###########################
y_true = pd.read_csv('scores.csv')['true']
y_logreg = pd.read_csv('scores.csv')['score_logreg']
y_svm = pd.read_csv('scores.csv')['score_svm']
y_knn = pd.read_csv('scores.csv')['score_knn']
y_tree = pd.read_csv('scores.csv')['score_tree']

roc_auc_logreg = roc_auc_score(y_true, y_logreg)
roc_auc_svm = roc_auc_score(y_true, y_svm)
roc_auc_knn = roc_auc_score(y_true, y_knn)
roc_auc_tree = roc_auc_score(y_true, y_tree)
print("logreg {} svm {} knn {} tree {}".format(roc_auc_logreg, roc_auc_svm, roc_auc_knn, roc_auc_tree))

###############task4##########

precision, recall, thresholds = precision_recall_curve(y_true, y_logreg)
c_logreg = max((recall>0.7)*precision)

precision, recall, thresholds = precision_recall_curve(y_true, y_svm)
c_svm = max((recall>0.7)*precision)

precision, recall, thresholds = precision_recall_curve(y_true, y_knn)
c_knn = max((recall>0.7)*precision)

precision, recall, thresholds = precision_recall_curve(y_true, y_tree)
c_tree = max((recall>0.7)*precision)

print('logreg {} svm {} knn {} tree {}'.format(c_logreg, c_svm, c_knn, c_tree))
