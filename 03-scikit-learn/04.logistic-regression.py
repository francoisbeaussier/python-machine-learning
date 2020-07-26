import numpy as np

plot_learning = True

# Logistic Regression is not a regression but a classification algorithm

class LogisticRegression(object):

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))
            self.cost.append(cost)
        return self

    def net_input(self,  X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1.0, 0.0)

# Loading the iris dataset
import pandas as pd

s = '/'.join(['https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data'])
print('Downloading from', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.head())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

# Feature scalling

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

lr = LogisticRegression(n_iter=100, eta=0.01).fit(X_std, y)

# Visualize the decision boundaries

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

# Plot the learning curve and decision regions

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax[0].plot(range(1, len(lr.cost) + 1), lr.cost)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log-likelihood')
ax[0].set_title('LogisticRegression')

ax[1].plot(range(1, len(lr.cost) + 1), lr.cost)
ax[1].set_xlabel('Petal length [std]')
ax[1].set_ylabel('Sepal length [std]')
ax[1].set_title('Decision regions')

plot_decision_regions(X_std, y, classifier=lr)

plt.show()