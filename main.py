import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
class linearRegression:

    def __init__(self, lr=0.001, n_iterations=100):
        self.lr = lr
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Plot init
        plt.ion()
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue')
        line, = ax.plot(X, self.predict(X), color='red')
        plt.title('Linear regression')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # plot update
            line.set_ydata(self.predict(X))
            fig.canvas.draw()
            fig.canvas.flush_events()

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def mse(y_predicted, y):
    return np.mean((y_predicted - y)**2)


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = linearRegression(lr=0.1, n_iterations=100)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

print(predicted)
print('mse: ', mse(predicted, y_test))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.scatter(X_test, y_test)
ax1.set_title("test values")

ax2.scatter(X_test, predicted)
ax2.set_title("preidcted values")

plt.show()
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
class linearRegression:

    def __init__(self, lr=0.001, n_iterations=100):
        self.lr = lr
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Plot init
        plt.ion()
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue')
        line, = ax.plot(X, self.predict(X), color='red')
        plt.title('Linear regression')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # plot update
            line.set_ydata(self.predict(X))
            fig.canvas.draw()
            fig.canvas.flush_events()

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def mse(y_predicted, y):
    return np.mean((y_predicted - y)**2)


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = linearRegression(lr=0.1, n_iterations=100)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

print(predicted)
print('mse: ', mse(predicted, y_test))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.scatter(X_test, y_test)
ax1.set_title("test values")

ax2.scatter(X_test, predicted)
ax2.set_title("preidcted values")

plt.show()
plt.waitforbuttonpress()