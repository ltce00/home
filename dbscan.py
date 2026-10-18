import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = [[1,1], [1,2], [2,1], [2,2], [7,7], [7,8], [8,7], [10,2]]
labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(X)

print("Labels:", labels)
plt.scatter([p[0] for p in X], [p[1] for p in X], c=labels)
plt.show()
