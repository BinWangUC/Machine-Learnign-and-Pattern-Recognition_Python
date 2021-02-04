# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:56:32 2020

@author: Bin Wang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt

diabetes_dataset=pd.read_csv("diabetes.csv")
diabetes_dataset.head()

results=[]
names=[]


#DataFlair - Get the features and labels
features=diabetes_dataset.loc[:,diabetes_dataset.columns!='Outcome'].values[:,1:]
labels=diabetes_dataset.loc[:,'Outcome'].values

x_train,x_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=42)

#KNN model and tuning
model=KNeighborsClassifier(2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

# model=KNeighborsClassifier()
# params={'n_neighbors':range(1,10)}
# params=[{'n_neighbors':range(1,10), 'p':[2],'leaf_size':[5,10,20,30,31,34,40,45],
#           'weights':['distance','uniform'],
#           'algorithm':['auto','ball_tree','kd_tree','brute']}]

# grs=GridSearchCV(model, param_grid=params)
# grs.fit(x_train,y_train)
# print("Best Hyper Parameters:",grs.best_params_)
# # model_best = grs.best_estimator_
# # y_pred = model_best.predict(x_test)
# y_pred=grs.predict(x_test)


kfold = model_selection.KFold(n_splits=20, shuffle=True,random_state=42)
cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append('KNN')
msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
print(msg)



# #SVM model and tunning
model= SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
# param_grid = [{'kernel': ['linear'],'class_weight':[None],
#                 'gamma': [0.0001],'C': [0.5, 0.1, 1, 5, 10]}]
# # param_grid = [{'C': [0.5, 0.1, 1, 5, 10], 'kernel': ['linear'], 'class_weight':['balanced']},
# #               {'C': [0.5, 0.1, 1, 5, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05,0.5],
# #                'kernel': ['rbf'], 'class_weight': ['balanced']}]
# grs = GridSearchCV(model, param_grid)
# grs.fit(x_train,y_train)
# print("Best Hyper Parameters:",grs.best_params_)
# model_best = grs.best_estimator_
# y_pred = model_best.predict(x_test)

# #Decision Tree Model and tuning
# model=DecisionTreeClassifier()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)

# params={'criterion':['gini','entropy'],'max_depth':range(1,10)}
# grs=GridSearchCV(model, param_grid=params)
# grs.fit(x_train,y_train)
# print("Best Hyper Parameters:", grs.best_params_)
# model=grs.best_estimator_
# y_pred=model.predict(x_test)

## plot comparision results
cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append('SVM')
msg = "%s: %f (%f)" % ('SVM', cv_results.mean(), cv_results.std())
print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## detailed accuracy explaination
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))
print("Matrix:",confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
