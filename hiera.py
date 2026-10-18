import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

X = [[1,2], [2,3], [3,2], [8,8], [9,8], [8,9]]
Z = linkage(X, method="complete")
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Point")
plt.ylabel("Distance")
plt.show()
