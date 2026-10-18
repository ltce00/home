from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]
model = Perceptron()
model.fit(X, y)
print("Prediction:", model.predict([[5, 3]]))
