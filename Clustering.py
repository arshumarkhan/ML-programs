import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = np.array([
    [2, 3], [3, 4], [4, 5],
    [8, 7], [7, 8], [9, 7],
    [1, 2], [2, 2], [3, 3]
])


kmeans = KMeans(n_clusters=2)


kmeans.fit(X)


labels = kmeans.labels_


centers = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Cluster Centers:", centers)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
plt.scatter(centers[:,0], centers[:,1], color='black', marker='X', s=200)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
