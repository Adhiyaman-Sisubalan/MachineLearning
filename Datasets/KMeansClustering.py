# -*- coding: utf-8 -*-

import HeaderFile as hf
import timeit

from sklearn.cluster import KMeans

data, features, target = hf.ReadData('winequality-red-clean-v2.csv')

X, y = hf.TransformData(features, target, resample=True, scale=True, pca=True, pca_plot=True)
print(hf.np.unique(y, return_counts=True))

wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig = hf.plt.figure(figsize=[10,10])
fig.subplots_adjust(hspace=0.7)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(range(1,11), wcss)
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('WCSS') 
ax1.set_title('Elbow Method for k-means Model')

kmeans = KMeans(n_clusters = 7, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
start_time = timeit.default_timer()
kmeans.fit(X)
end_time = timeit.default_timer()
elapsed_time = end_time-start_time

y_kmeans = kmeans.predict(X)

X_arr = X.to_numpy()
ax2.scatter(X_arr[y_kmeans == 0, 0], X_arr[y_kmeans == 0, 1], s = 5, c = 'black', label = 'Cluster 0' )
ax2.scatter(X_arr[y_kmeans == 1, 0], X_arr[y_kmeans == 1, 1], s = 5, c = 'blue', label = 'Cluster 1' )
ax2.scatter(X_arr[y_kmeans == 2, 0], X_arr[y_kmeans == 2, 1], s = 5, c = 'green', label = 'Cluster 2' )
ax2.scatter(X_arr[y_kmeans == 3, 0], X_arr[y_kmeans == 3, 1], s = 5, c = 'purple', label = 'Cluster 3' )
ax2.scatter(X_arr[y_kmeans == 4, 0], X_arr[y_kmeans == 4, 1], s = 5, c = 'cyan', label = 'Cluster 4' )
ax2.scatter(X_arr[y_kmeans == 5, 0], X_arr[y_kmeans == 5, 1], s = 5, c = 'brown', label = 'Cluster 5' )
ax2.scatter(X_arr[y_kmeans == 6, 0], X_arr[y_kmeans == 6, 1], s = 5, c = 'red', label = 'Cluster 6' )

ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='yellow', label='Centroids')
ax2.set_xlabel('Principle Component 1')
ax2.set_ylabel('Principle Component 2') 
ax2.set_title('K-means\nWine Cluster')
ax2.xaxis.set_ticklabels(['4','6','8','10','12','14'])
ax2.legend(loc='best')