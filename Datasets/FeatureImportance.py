# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re as re
import timeit as timeit

from sklearn.ensemble import ExtraTreesClassifier


def SetAxisProperties(title, xticklabels, line):
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Average Feature Importance Scores (1000 Runs)")
    plt.xticks(np.arange(11), xticklabels, rotation=45)
    plt.yticks(np.arange(0.07, 0.17, step=0.01))
    plt.legend(handles=[line], labels=[line.get_label()], loc="best")
    plt.tight_layout()


def ReplaceUpper(matchObj):

    return matchObj.group(0).upper()


wine_data = pd.read_csv("winequality-red.csv", delimiter=";")
wine_data_quality_count = wine_data.groupby("quality").quality.count()
column_names = wine_data.columns.values.tolist()
feature_names = column_names[0:11]
target_name = column_names[-1]
features = wine_data[feature_names]
target = wine_data[target_name]

number_iterations = 1000
feature_importance_scores = np.zeros((number_iterations, 11))
start_time = timeit.default_timer()
for i in range(number_iterations):
    etf = ExtraTreesClassifier(n_estimators=10)
    etf.fit(features, target)
    feature_importance_scores[i] = etf.feature_importances_
feature_importance_scores_mean = np.mean(feature_importance_scores, axis=0)
feature_importance_scores_std = np.std(feature_importance_scores, axis=0)
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
fism_list = list(feature_importance_scores_mean)
min_importance_score = np.mean(feature_importance_scores_mean)
selected_features = [feature_names[fism_list.index(f)] for f in feature_importance_scores_mean if f > min_importance_score]

sorting_order = feature_importance_scores_mean.argsort()
feature_importance_scores_mean_desc = feature_importance_scores_mean[sorting_order[::-1]]
feature_importance_scores_std_desc = feature_importance_scores_std[sorting_order[::-1]]
feature_names_desc = np.array(feature_names)[sorting_order[::-1]].tolist()

feature_names_caps = [re.sub("(?<!\S)[a-z]", ReplaceUpper, name) for name in feature_names]
feature_names_caps_desc = [re.sub("(?<!\S)[a-z]", ReplaceUpper, name) for name in feature_names_desc]

plt.figure()
plt.errorbar(np.arange(11), feature_importance_scores_mean, yerr=feature_importance_scores_std, capsize=10, fmt="ob-")
min_line = plt.axhline(y=min_importance_score, color='r', linestyle='-.', label="Mean of Importance Scores")
SetAxisProperties("Feature Importance Rankings (Pre-Sort)", feature_names_caps, min_line)

plt.figure()
plt.errorbar(np.arange(11), feature_importance_scores_mean_desc, yerr=feature_importance_scores_std_desc, capsize=10, fmt="ob-")
min_line = plt.axhline(y=min_importance_score, color='r', linestyle='-.', label="Mean of Importance Scores")
SetAxisProperties("Feature Importance Rankings (Post-Sort)", feature_names_caps_desc, min_line)
