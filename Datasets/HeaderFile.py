# -*- coding: utf-8 -*-

import imblearn.over_sampling as os
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statistics as stats

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def ReadData(path, delimiter=','):
    data = pd.read_csv(path, delimiter)
    data_shape = data.shape
    columns = data.columns.values.tolist()
    features = columns[0:data_shape[1]-1]
    target = columns[-1]
    features = data[features]
    target = data[target]

    return data, features, target


def replacing(df, digitizebin):
    which_bin_dict = {}
    mean_bin_dict = {}
    replaced = []
    for i in range(0, len(digitizebin)):
        if digitizebin[i] in which_bin_dict:
            which_bin_dict[digitizebin[i]].append(df.iloc[i])
        else:
            which_bin_dict[digitizebin[i]] = [df.iloc[i]]
    for k, v in which_bin_dict.items():
        mean_bin_dict.update({k: stats.mean(v)})
    for j in range(0, len(digitizebin)):
        r = mean_bin_dict[digitizebin[j]]
        replaced.append(r)

    return replaced


def binning(dataframe, no_of_bins):
    dfl = list(dataframe.columns)
    for x in dfl:
        bins = np.linspace(dataframe[x].min(), dataframe[x].max(), no_of_bins)
        m = np.digitize(dataframe[x], bins=bins)
        dataframe[x] = replacing(dataframe[x], m)
        #print(x)
        #print(bins)
        #print(dataframe[x])

    return dataframe


def TransformData(features, target, resample=False, scale=False, pca=False, pca_plot=False):
    X = features
    y = target
    if resample is not False:
        over_sampler = os.BorderlineSMOTE(n_jobs=-1, random_state=42)
        #over_sampler = os.RandomOverSampler(random_state=42)
        #over_sampler = os.SMOTE(n_jobs=-1, random_state=42)
        #over_sampler = os.SVMSMOTE(n_jobs=-1, random_state=42)
        X, y = over_sampler.fit_resample(X, y)
    if scale is not False:
        standard_scaler = StandardScaler()
        X_scaled = standard_scaler.fit_transform(X)
        X = pd.DataFrame(data=X_scaled)
    if pca is not False:
        pca_obj = PCA()
        principal_components = pca_obj.fit_transform(X)
        print(pca_obj.explained_variance_ratio_)
        print(pca_obj.explained_variance_ratio_.sum())
        X = pd.DataFrame(data=principal_components)
        if pca_plot is not False:
            Plot2D("bar", np.arange(pca_obj.n_components_), 100*pca_obj.explained_variance_ratio_,
                   title='Relative Information Content of PCA Components',
                   xlabel="PCA Component Numbers",
                   ylabel="PCA Component Variance %s")

    return X, y


def ComputeScores(estimator, params, xtrain, xtest, ytrain, ytest):
    grid_size = 1
    for p in params:
        grid_size *= len(params[p])
    cv_scores = np.zeros(grid_size)
    train_scores = np.zeros(grid_size)
    test_scores = np.zeros(grid_size)
    param_keys = list(params.keys())
    param_values = list(params.values())
    index = 0
    for itr in it.product(*param_values):
        est_params = {param_keys[i]: itr[i] for i in range(len(itr))}
        estimator.set_params(**est_params)
        print(estimator.get_params)
        scores = cross_val_score(estimator, X=xtrain, y=ytrain, cv=10, error_score=0, n_jobs=-1, scoring="accuracy")
        cv_scores[index] = scores.mean()
        estimator.fit(xtrain, ytrain)
        train_scores[index] = estimator.score(xtrain, ytrain)
        test_scores[index] = estimator.score(xtest, ytest)
        index += 1

    return cv_scores, train_scores, test_scores


def Plot2D(kind, xvalues, yvalues, linelabels=None, title=None, xlabel=None, ylabel=None, xticks=None, yticks=None):
    '''
    yvalues, linelabels: Must be lists & |linelabels| <= |yvalues|
    xticks, yticks: Must be dictionaries.
    '''
    plt.figure()
    if kind == "bar":
        plt.bar(xvalues, yvalues)
    elif kind == "line":
        for i in range(len(yvalues)):
            line = plt.plot(np.arange(len(xvalues)), yvalues[i], linestyle='--', marker='o')
            if linelabels is not None:
                line[0].set_label(linelabels[i])
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(**xticks)
    if yticks is not None:
        plt.yticks(**yticks)
    if linelabels is not None:
        plt.legend()
    plt.tight_layout()


def PlotConfusionMatrix(cmleft, cmright, y_hat_acc, y_pred_acc, suptitle=None, ticklabels=None, color="jet"):
    plt.figure()
    plt.suptitle(suptitle)
    ax_left = plt.subplot(121)
    ax_right = plt.subplot(122)
    sns.heatmap(cmleft, annot=True, ax=ax_left, cbar_kws={"shrink": 0.5}, cmap=color, fmt='g', square=True)
    sns.heatmap(cmright, annot=True, ax=ax_right, cbar_kws={"shrink": 0.5}, cmap=color, fmt='g', square=True)
    SetCmAxisProperties(ax_left, r"Train Data (Accuracy$\approx{%.3f}$)" % (y_hat_acc), ticklabels)
    SetCmAxisProperties(ax_right, r"Test Data (Accuracy$\approx{%.3f}$)" % (y_pred_acc), ticklabels)


def SetCmAxisProperties(ax, title, ticklabels=None):
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if ticklabels is not None:
        ax.xaxis.set_ticklabels(ticklabels)
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        ax.yaxis.set_ticklabels(ticklabels)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
