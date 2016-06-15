## Try to solve the salesman problem with kmeans

import numpy as np
import pylab as plt

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

## Parameters of the models
km_sizes = 4
min_point_to_heuristique = 10
nb_cities = 300

## Generate some cities
centers = [[1, 1], [3, 1.2], [2.5, 3.4], [1.2,2.6]]
country, colors = make_blobs(n_samples=nb_cities, centers=centers, cluster_std=1, random_state=42)

## Visualize our country
plt.scatter(country[:,0],country[:,1],c=colors)
plt.show()



#### Todo : should use a recursive  iteration of kmeans fit
## Use a Tree data structure to keep trak of labels inside clusters
## Solve

km_0 = KMeans(init='k-means++', n_clusters=km_sizes, n_init=10)
km_0.fit(country)
plt.scatter(country[:,0],country[:,1],c=km_0.labels_)
plt.show()

## keep an array of cluster and sub cluster
cluster_level = np.hstack([country,km_0.labels_.reshape(-1,1)])



## SO now, redo inside each cluster

## now make cluster inside clusters

for current_cluster in set(km_0.labels_):
  ## new sub city
  cluster_level = np.hstack([cluster_level,np.zeros((300,1))])
  sub_urbs = country[np.where(km_0.labels_==current_cluster)]
  sub_km = KMeans(init='k-means++', n_clusters=km_sizes, n_init=10)
  sub_km.fit(sub_urbs)
  cluster_level[:,-1][np.where(km_0.labels_==0)] = sub_km.labels_


## Once the cluster is small enough ( less than min_point_to_heuristique), solve it 
## with brute force


## When a cluster is solve, move to the nearest cluster insde the parent cluster
