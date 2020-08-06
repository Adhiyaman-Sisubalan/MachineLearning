# -*- coding: utf-8 -*-

import HeaderFile as hf
import timeit

from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib.lines import Line2D

data, features, target = hf.ReadData('winequality-red-clean-v2.csv')

X, y = hf.TransformData(features, target, resample=True, scale=True, pca=True, pca_plot=True)
print(hf.np.unique(y, return_counts=True))

Z = linkage(X, 'ward')

hf.plt.figure(figsize=(25,10))
hf.plt.title('Red Wine Hierarchical Clustering Dendrogram')
hf.plt.xlabel('Quality')
hf.plt.ylabel('Distance')
dendrogram(
        Z, truncate_mode='lastp', p=150,
        leaf_rotation=90.,
        leaf_font_size=8.)

max_d = 88
hf.plt.axhline(y=max_d, c='k')

clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
start_time = timeit.default_timer()
clustering.fit(X)
end_time = timeit.default_timer()
elapsed_time = end_time-start_time

X_plot = preprocessing.MinMaxScaler().fit_transform(X)

hf.plt.figure(figsize=(25,10))
colors = 'rgb'
for i in range(X.shape[0]):
    hf.plt.text(X_plot[i,0], 
                X_plot[i,1], 
                str(clustering.labels_[i]), 
                color=colors[clustering.labels_[i]],
                fontdict={'weight':'bold','size':9})

hf.plt.title('Red Wine Hierarchical Clustering')
hf.plt.xlabel('Principal Component 1')
hf.plt.ylabel('Principal Component 2') 

legend_elements = [Line2D([0],[0],marker='o',color='w',label='Cluster 0',markerfacecolor='r',markersize=12),
                   Line2D([0],[0],marker='o',color='w',label='Cluster 1',markerfacecolor='g',markersize=12),
                   Line2D([0],[0],marker='o',color='w',label='Cluster 2',markerfacecolor='b',markersize=12)]
hf.plt.legend(handles=legend_elements,loc='best')