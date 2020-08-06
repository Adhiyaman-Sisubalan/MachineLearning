# -*- coding: utf-8 -*-

import timeit as timeit

import HeaderFile as hf

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from subprocess import call

data, features, target = hf.ReadData("winequality-red-clean-v2.csv")

X, y = hf.TransformData(features, target, resample=True, scale=True, pca=True, pca_plot=True)
print(hf.np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

criterions = ["gini", "entropy"]
max_depths = list(range(2, 21))
search_pairs = list(hf.it.product(criterions, max_depths))
cv_scores, train_scores, test_scores = hf.ComputeScores(DecisionTreeClassifier(),
                                                        {"criterion": criterions, "max_depth": max_depths},
                                                        x_train, x_test, y_train, y_test)

hf.Plot2D("line", xvalues=hf.np.arange(len(search_pairs)), yvalues=[cv_scores, train_scores, test_scores],
          linelabels=["CV Means", "Train", "Test"],
          title="Accuracy Scores with Different Search Parameters",
          xlabel="Search Pairs",
          ylabel="Accuracy Scores",
          xticks={"ticks": hf.np.arange(len(search_pairs)), "labels": search_pairs, "rotation": 45})

#dt = DecisionTreeClassifier()
dt = DecisionTreeClassifier(criterion="gini", max_depth=7)
start_time = timeit.default_timer()
dt.fit(x_train, y_train)
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
y_hat = dt.predict(x_train)
y_hat_accuracy = accuracy_score(y_train, y_hat)
y_pred = dt.predict(x_test)
y_pred_accuracy = accuracy_score(y_test, y_pred)
cl_rp = classification_report(y_test, y_pred)

qualities = hf.np.unique(y)
hf.PlotConfusionMatrix(confusion_matrix(y_train, y_hat), confusion_matrix(y_test, y_pred),
                       suptitle="Decision Tree Confusion Matrix (criterion = \"gini\", max_depth = 7)",
                       ticklabels=qualities,
                       color="Blues",
                       y_hat_acc=y_hat_accuracy,
                       y_pred_acc=y_pred_accuracy)

'''
tree.export_graphviz(dt, out_file="tree.dot", class_names=True, feature_names=x_train.columns)
call(["dot", "-Tpng", "tree.dot", "-o", "tree.png", "-Gdpi=600"])
'''
