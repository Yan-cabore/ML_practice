import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

plt.scatter(X[:, 0], X[:, 1])

plt.xlabel("Sepal Lenght")
plt.ylabel("Sepal Widht")
plt.title("Iris Scatter Plot")

plt.show()


