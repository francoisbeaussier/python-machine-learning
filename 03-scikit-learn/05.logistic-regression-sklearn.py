from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Labels:', np.unique(y))

from sklearn.model_selection import train_test_split

# train_test_split also shuffles the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# stratify=y means we preserve the proportions of labels across input, train and test labels

print("Labels count in y:", np.bincount(y))
print("Labels count in y_train:", np.bincount(y_train))
print("Labels count in y_test:", np.bincount(y_test))

# Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training a perceptron

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)
misclassified = (y_test != y_pred).sum()
print(f'Misclassified: {misclassified}')

print(f'Accuracy: {lr.score(X_test_std, y_test)}')

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(f'Predict: {lr.predict_proba(X_test_std[:3, :])}')

# Argmax can be used to find the highest probability

print(f'Predict: {lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)}')

# Note: to compute a single prediction, we have to reshape to add one dimension
print(f'Predict: {lr.predict_proba(X_test_std[0, :].reshape(1, -1)).argmax(axis=1)}')

# Plot decision regions

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
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
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=[colors[idx]], marker=markers[idx], label=cl, edgecolor='black')

    # Highlight test examples

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [std]')
plt.ylabel('sepal length [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()