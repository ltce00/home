import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
X_new = PCA(n_components=2).fit_transform(X)

print("Reduced shape:", X_new.shape)
plt.scatter(X_new[:, 0], X_new[:, 1], c=iris.target)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
