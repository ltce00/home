import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [[1,2], [2,3], [3,2], [8,8], [9,8], [8,9]]
k = KMeans(n_clusters=2, n_init=10).fit(X)

print("Labels:", k.labels_)
plt.scatter([p[0] for p in X], [p[1] for p in X], c=k.labels_)
plt.show()
