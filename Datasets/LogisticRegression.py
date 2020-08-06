# -*- coding: utf-8 -*-

import timeit as timeit

import HeaderFile as hf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data, features, target = hf.ReadData("winequality-red-clean-v2.csv")

X, y = hf.TransformData(features, target, resample=True, scale=True, pca=True, pca_plot=True)
print(hf.np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_scores, train_scores, test_scores = hf.ComputeScores(LogisticRegression(solver="lbfgs", max_iter=3000, tol=1e-6, multi_class="multinomial"),
                                                        {"C": c_values},
                                                        x_train, x_test, y_train, y_test)

hf.Plot2D("line", xvalues=c_values, yvalues=[cv_scores, train_scores, test_scores],
          linelabels=["CV Means", "Train", "Test"],
          title="Accuracy Scores with Increasing C-values",
          xlabel="C-values",
          ylabel="Accuracy Scores",
          xticks={"ticks": hf.np.arange(len(c_values)), "labels": c_values})

#log_reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
log_reg = LogisticRegression(solver="lbfgs", C=1, max_iter=3000, tol=1e-6, multi_class="multinomial")
start_time = timeit.default_timer()
log_reg.fit(x_train, y_train)
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
y_hat = log_reg.predict(x_train)
y_hat_accuracy = accuracy_score(y_train, y_hat)
y_pred = log_reg.predict(x_test)
y_pred_accuracy = accuracy_score(y_test, y_pred)
cl_rp = classification_report(y_test, y_pred)

qualities = hf.np.unique(y)
hf.PlotConfusionMatrix(confusion_matrix(y_train, y_hat), confusion_matrix(y_test, y_pred),
                       suptitle="Logistic Regression Confusion Matrix (C = 0.1)",
                       ticklabels=qualities,
                       color="Reds",
                       y_hat_acc=y_hat_accuracy,
                       y_pred_acc=y_pred_accuracy)
