# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:56:20 2020

@author: Bin WANG
"""

#import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

parkinson_dataset=pd.read_csv("parkinsons.data")
parkinson_dataset.head()

#Get the features and labels
features=parkinson_dataset.loc[:,parkinson_dataset.columns!='status'].values[:,1:]
labels=parkinson_dataset.loc[:,'status'].values

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=42)

results=[]
names=[]

#Decision Tree
model=DecisionTreeClassifier()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# params={'max_leaf_nodes':[1000,200,30,14,15,16,17,None], 'criterion':['gini','entropy'],'max_depth':range(1,10)}
# # 'criterion':['gini','entropy'],'max_depth':range(1,10)
# grs=GridSearchCV(model, param_grid=params)
# grs.fit(x_train,y_train)
# print("Best Hyper Parameters:", grs.best_params_)
# model=grs.best_estimator_
# y_pred=model.predict(x_test)
## collect result
kfold = model_selection.KFold(n_splits=20, shuffle=True,random_state=42)
cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append('Decision Tree')
msg = "%s: %f (%f)" % ('Decision Tree', cv_results.mean(), cv_results.std())
print(msg)


# #SVM model and tunning
# model= SVC()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# # param_grid = {'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05,0.5],'class_weight': ['balanced'],'kernel': ['rbf']}
#                 # 'gamma': [0.0001],'C': [0.5, 0.1, 1, 5, 10]}]
# param_grid = [{'C': [0.5, 0.1, 1, 5, 10], 'kernel': ['linear'], 'class_weight':['balanced']},
#               {'C': [0.5, 0.1, 1, 5, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05,0.5],
#                 'kernel': ['linear'], 'class_weight': ['balanced']}]
# grs = GridSearchCV(model, param_grid)
# grs.fit(x_train,y_train)
# print("Best Hyper Parameters:",grs.best_params_)
# model_best = grs.best_estimator_
# y_pred = model_best.predict(x_test)

cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append('SVM')
msg = "%s: %f (%f)" % ('SVM', cv_results.mean(), cv_results.std())
print(msg)

# Naive Bayesian
model = GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
# param_grid = {'var_smoothing':[1e-010,1e-09,1e-09,1e-08,1e-07,1e-05,1e-03]}
# grs = GridSearchCV(model, param_grid)
# grs.fit(x_train,y_train)
# print("Best Hyper Parameters:",grs.best_params_)
# model_best = grs.best_estimator_
# y_pred = model_best.predict(x_test)


## plot comparision results
cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
names.append('Naive Bayesian')
msg = "%s: %f (%f)" % ('Naive Bayesian', cv_results.mean(), cv_results.std())
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
