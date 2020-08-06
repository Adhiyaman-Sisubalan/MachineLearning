# -*- coding: utf-8 -*-
'''
Fitting the various supervised Machine Learning Model with raw dataset to obtain a baseline for comparisons
with improvements.
'''
# Import the required Python Packages
import pandas as pd

import HeaderFile as hf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Reading the data from the data set
wine_data = pd.read_csv("winequality-red.csv", delimiter=";")
x = wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']]
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Logistic Regression
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)
y_pred_test = model_lr.predict(X_test)
y_pred_train = model_lr.predict(X_train)
as_test = accuracy_score(y_test, y_pred_test)
as_train = accuracy_score(y_train, y_pred_train)

hf.PlotConfusionMatrix(confusion_matrix(y_train, y_pred_train), confusion_matrix(y_test, y_pred_test),
                       suptitle="Logistic Regression Confusion Matrix (C = 1)",
                       color="Reds",
                       y_hat_acc=as_train,
                       y_pred_acc=as_test)

# KNN Classifier
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
y_pred_test = model_knn.predict(X_test)
y_pred_train = model_knn.predict(X_train)
as_test = accuracy_score(y_test, y_pred_test)
as_train = accuracy_score(y_train, y_pred_train)

hf.PlotConfusionMatrix(confusion_matrix(y_train, y_pred_train), confusion_matrix(y_test, y_pred_test),
                       suptitle="KNN Confusion Matrix (n_neighbors=5)",
                       color="Greens",
                       y_hat_acc=as_train,
                       y_pred_acc=as_test)


# DecisionTree Classifier
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_test = model_dt.predict(X_test)
y_pred_train = model_dt.predict(X_train)
as_test = accuracy_score(y_test, y_pred_test)
as_train = accuracy_score(y_train, y_pred_train)

hf.PlotConfusionMatrix(confusion_matrix(y_train, y_pred_train), confusion_matrix(y_test, y_pred_test),
                       suptitle="Decision Tree Confusion Matrix (criterion='gini', max_depth=None)",
                       color="Blues",
                       y_hat_acc=as_train,
                       y_pred_acc=as_test)
