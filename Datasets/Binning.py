# -*- coding: utf-8 -*-
'''
Fitting Machine Learning Model with raw dataset to obtain a baseline for comparisons
with improvements.
'''
# Import the required Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import HeaderFile as hf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# Returns train and test accuracies arrays
def perform_binning(numberofbins, str):
    test_accuracy = []
    train_accuracy = []
    bestbin = 0
    ta_previous = 0

    for b in numberofbins:
        wine_data = pd.read_csv("winequality-red.csv", delimiter=";")
        scalar = StandardScaler()
        wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                   'pH', 'sulphates', 'alcohol']] = scalar.fit_transform(
           wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol']])
        wine_data_x = wine_data.iloc[:, :11]
        wine_data_y = wine_data['quality']
        wine_data_x_binned = hf.binning(wine_data_x, b)

        X_train, X_test, y_train, y_test = train_test_split(wine_data_x_binned, wine_data_y, random_state=42)
        if str == "lr":
            model = LogisticRegression(random_state=42)
        elif str == "knn":
            model = KNeighborsClassifier()
        elif str == "dt":
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        y_pred_train = model.predict(X_train)
        ta = accuracy_score(y_test, y_pred_test)
        test_accuracy.append(ta)
        train_accuracy.append(accuracy_score(y_train, y_pred_train))
        if ta > ta_previous:
            ta_previous = ta
            bestbin = b

    fig = plt.figure(figsize=[10, 10])
    fig.subplots_adjust(hspace=0.7)
    plt.plot(numberofbins, test_accuracy, color='r', label="Test Accuracy")
    plt.plot(numberofbins, train_accuracy, '--', color='g', label="Train Accuracy")
    if str == "lr":
        plt.title("Effect of Binning on the Accuracy of Logistic Regression Machine Learning Model\n" +
                  "Bins with highest accuracy = {0}".format(bestbin), fontsize=15)
    elif str == "knn":
        plt.title("Effect of Binning on the Accuracy of KNN Classifier Machine Learning Model\n" +
                  "Bins with highest accuracy = {0}".format(bestbin), fontsize=15)
    elif str == "dt":
        plt.title("Effect of Binning on the Accuracy of Decision Tree Machine Learning Model\n" +
                  "Bins with highest accuracy = {0}".format(bestbin), fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.xlabel("Number of Bins", fontsize=15)
    plt.legend()


bins = np.linspace(1, 10, 10)
perform_binning(bins, "lr")
perform_binning(bins, "knn")
perform_binning(bins, "dt")
