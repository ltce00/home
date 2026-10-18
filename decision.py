from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
m = DecisionTreeClassifier(max_depth=3).fit(X, y)

print("Prediction:", m.predict([X[0]]))
print("Accuracy:", m.score(X, y))
plot_tree(m, filled=True)
plt.show()
