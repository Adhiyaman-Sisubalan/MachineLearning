# -*- coding: utf-8 -*-

import HeaderFile as hf
import timeit

from sklearn.cluster import DBSCAN

data, features, target = hf.ReadData('winequality-red-clean-v2.csv')

X, y = hf.TransformData(features, target, resample=True, scale=True, pca=True, pca_plot=True)
print(hf.np.unique(y, return_counts=True))
X_arr = X.to_numpy()

dbscan = DBSCAN(eps=2.2, min_samples=10)

start_time = timeit.default_timer()
clusters = dbscan.fit_predict(X)
end_time = timeit.default_timer()
elapsed_time = end_time-start_time

n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
print(hf.np.unique(clusters, return_counts=True))

hf.plt.figure(figsize=(25,10))
hf.plt.scatter(X_arr[:,0], X_arr[:,1], c=clusters, s=20, cmap="rainbow")
hf.plt.title('DBSCAN\n (Number of clusters = {0}, Number of Noise sample = {1})'.format(n_clusters_ , n_noise_))
hf.plt.xlabel('Principal Component 1')
hf.plt.ylabel('Principal Component 2') 
hf.plt.legend()
