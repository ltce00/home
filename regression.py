from sklearn.linear_model import LinearRegression, LogisticRegression

# Part A: Linear Regression
X, y = [[1], [2], [3], [4], [5]], [20, 30, 40, 50, 60]
m = LinearRegression().fit(X, y)
print("Predicted marks:", m.predict([[6]])[0])

# Part B: Logistic Regression
X, y = [[20], [30], [40], [60], [70], [80]], [0, 0, 0, 1, 1, 1]
m = LogisticRegression().fit(X, y)
print("Prediction:", m.predict([[65]])[0])
print("Accuracy:", m.score(X, y))
