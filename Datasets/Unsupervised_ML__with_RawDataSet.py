# -*- coding: utf-8 -*-
'''
Training the various unspervised machine learning model with the raw dataset
'''

# Importing the necessary Python Packages
import matplotlib.pyplot as plt
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data Preparation for model fitting, extracting only the features for unsupervised learning
wine_data = pd.read_csv("winequality-red.csv", delimiter=";")
x = wine_data.iloc[:, :11].values
y = wine_data.iloc[:, -1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Visualising the clusters
fig = plt.figure(figsize=[10, 10])
fig.subplots_adjust(hspace=0.7)
ax = fig.add_subplot(2, 1, 1)

ax.scatter(X_scaled[y == 3, 0], X_scaled[y == 3, 1], s=5, c="red", label='Quality 3')
ax.scatter(X_scaled[y == 4, 0], X_scaled[y == 4, 1], s=5, c="blue", label='Quality 4')
ax.scatter(X_scaled[y == 5, 0], X_scaled[y == 5, 1], s=5, c="green", label='Quality 5')
ax.scatter(X_scaled[y == 6, 0], X_scaled[y == 6, 1], s=5, c="purple", label='Quality 6')
ax.scatter(X_scaled[y == 7, 0], X_scaled[y == 7, 1], s=5, c="black", label='Quality 7')
ax.scatter(X_scaled[y == 8, 0], X_scaled[y == 8, 1], s=5, c="brown", label='Quality 8')

ax.set_title("Scatterplot for Fixed Acidity Against Volatile Acidity Without Engineering", fontsize=15)
ax.set_xlabel('Fixed Acidity', fontsize=15)
ax.set_ylabel('Volatile Acidity', fontsize=15)
ax.legend()

print("\n****************** k-means ******************")
kmc = KMeans(n_clusters=6, init="k-means++", max_iter=300, n_init=10, random_state=0)
print("*Model Details*")
print(kmc)
kmc.fit(X_scaled)
y_kmeans = kmc.fit_predict(X_scaled)

# Finding the optimum number of cluster for k-means classification
wcss = []
# Trying kmeans for k = 1 to k = 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the respective visualization
fig = plt.figure(figsize=[10, 10])
fig.subplots_adjust(hspace=0.7)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
fig2 = plt.figure(figsize=[10, 10])
fig2.subplots_adjust(hspace=0.7)
ax3 = fig2.add_subplot(2, 1, 1)
fig3 = plt.figure(figsize=[10, 10])
fig3.subplots_adjust(hspace=0.7)
ax4 = fig3.add_subplot(2, 1, 1)

ax1.plot(range(1, 11), wcss)
ax1.set_xlabel('Number of Clusters', fontsize=15)
ax1.set_ylabel('WCSS', fontsize=15)
ax1.set_title('The Elbow Method for k-means Machine Learning Model', fontsize=15)

# The optimal number of cluster is 3
kmc_3 = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmc_3.fit(X_scaled)
y_kmeans_3 = kmc_3.fit_predict(X_scaled)

# Visualising the clusters
ax2.scatter(X_scaled[y_kmeans_3 == 0, 0], X_scaled[y_kmeans_3 == 0, 1], s=5, c="red", label='Cluster 0')
ax2.scatter(X_scaled[y_kmeans_3 == 1, 0], X_scaled[y_kmeans_3 == 1, 1], s=5, c="blue", label='Cluster 1')
ax2.scatter(X_scaled[y_kmeans_3 == 2, 0], X_scaled[y_kmeans_3 == 2, 1], s=5, c="green", label='Cluster 2')

# Plotting of the centroids of the clusters
ax2.scatter(kmc_3.cluster_centers_[:, 0], kmc_3.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
ax2.set_xlabel('Fixed Acidity', fontsize=15)
ax2.set_ylabel('Volatile Acidity', fontsize=15)
ax2.set_title('k-means\nWine Cluster based on Wine Features\n (n_cluster = 3)', fontsize=15)
ax2.xaxis.set_ticklabels(['4', '6', '8', '10', '12', '14'])

print("\n****************** Hierarchical Clustering******************")
clustering = AgglomerativeClustering(linkage="ward")
print("*Model Details*")
print(clustering)
clustering.fit(X_scaled)
# set cut-off to 150 cluster merges
# max_d = 7.08                # max_d as in max_distance

Z = linkage(X_scaled, 'ward')

ax4.set_title('Wine Quality Hierarchical Clustering Dendrogram', fontsize=15)
ax4.set_xlabel('Quality', fontsize=15)
ax4.set_ylabel('Distance', fontsize=15)
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters the both part won't be shown
    p=150,                # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8., ax=ax4      # font size for the x axis labels
)
# ax4.axhline(y=max_d, c='k')

print("\n****************** DBSCAN ******************")

# X_scaled = scaler.fit_transform(x)
dbscan = DBSCAN()
print("*Model Details*")
print(dbscan)
clusters = dbscan.fit_predict(X_scaled)
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)

# print(np.unique(clusters))
print(f'Number of clusters = {n_clusters_}')
print(f'Number of noise sample = {n_noise_}')

# Plotting the cluster assignment
ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, s=5, cmap="plasma")
ax3.set_title('DBSCAN\n (Number of clusters = {0}, Number of Noise sample = {1})'.format(n_clusters_, n_noise_), fontsize=15)
ax3.set_xlabel("Fixed Acidity", fontsize=15)
ax3.set_ylabel("Volatile Acidity", fontsize=15)

ax2.legend()
