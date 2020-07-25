import numpy as np

plot_learning = True

class Adaline(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.cost = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.errors =[]

        for _ in range(self.n_iter):
            errors = 0
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self,  X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1 , -1)

# Loading the iris dataset
import pandas as pd

s = '/'.join(['https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data'])
print('Downloading from', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.head())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

# Training adaline

import matplotlib.pyplot as plt

if plot_learning:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    a1 = Adaline(eta=0.01, n_iter=10).fit(X, y)

    ax[0].plot(range(1, len(a1.cost) + 1), np.log10(a1.cost), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    a2 = Adaline(eta=0.0001, n_iter=10).fit(X, y)
    
    ax[1].plot(range(1, len(a2.cost) + 1), a2.cost, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()

# Let's apply feature scalling to help the learning process

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

a3 = Adaline(n_iter=15, eta=0.01).fit(X_std, y)

# Visualize the decision boundaries

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

# Beautiful learning curve!
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax[0].plot(range(1, len(a3.cost) + 1), np.log10(a3.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Feature scaling - Learning rate 0.01')

plot_decision_regions(X_std, y, classifier=a3)
plt.title('Adaline with Feature Scaling')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()