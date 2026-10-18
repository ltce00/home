from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = load_iris(return_X_y=True)
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3)
m = RandomForestClassifier(n_estimators=10).fit(X1, y1)
p = m.predict(X2)

print("Accuracy:", accuracy_score(y2, p))
print("Confusion Matrix:\n", confusion_matrix(y2, p))
