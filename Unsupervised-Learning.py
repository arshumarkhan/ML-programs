import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11],
    [8, 2], [10, 2], [9, 3]
])

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

labels = kmeans.labels_

centers = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Cluster Centers:", centers)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
plt.scatter(centers[:,0], centers[:,1], color='black', marker='X', s=200)
plt.title("Unsupervised Learning - KMeans Clustering")
plt.show()
