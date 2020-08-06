# -*- coding: utf-8 -*-

import timeit as timeit

import HeaderFile as hf

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data, features, target = hf.ReadData("winequality-red-clean-v2.csv")

X, y = hf.TransformData(features, target, resample=True, scale=True, pca=True, pca_plot=True)
print(hf.np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

k_values = [i for i in range(1, 31, 2)]
cv_scores, train_scores, test_scores = hf.ComputeScores(KNeighborsClassifier(),
                                                        {"n_neighbors": k_values},
                                                        x_train, x_test, y_train, y_test)

hf.Plot2D("line", xvalues=k_values, yvalues=[cv_scores, train_scores, test_scores],
          linelabels=["CV Means", "Train", "Test"],
          title="Accuracy Scores with Increasing k-values",
          xlabel="k-values",
          ylabel="Accuracy Scores",
          xticks={"ticks": hf.np.arange(len(k_values)), "labels": k_values})

#knn_model = KNeighborsClassifier()
knn_model = KNeighborsClassifier(n_neighbors=11)
start_time = timeit.default_timer()
knn_model.fit(x_train, y_train)
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
y_hat = knn_model.predict(x_train)
y_hat_accuracy = accuracy_score(y_train, y_hat)
y_pred = knn_model.predict(x_test)
y_pred_accuracy = accuracy_score(y_test, y_pred)
cl_rp = classification_report(y_test, y_pred)

qualities = hf.np.unique(y)
hf.PlotConfusionMatrix(confusion_matrix(y_train, y_hat), confusion_matrix(y_test, y_pred),
                       suptitle="KNN Confusion Matrix (k = 11)",
                       ticklabels=qualities,
                       color="Greens",
                       y_hat_acc=y_hat_accuracy,
                       y_pred_acc=y_pred_accuracy)
