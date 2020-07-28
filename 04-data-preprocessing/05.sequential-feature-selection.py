import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)
df_wine.columns=[
    'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoi phenols', 
    'proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head())

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Standard scaling

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Sequencial feature selection (greedy)

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
        dim = X_train.shape[1]
        self.indices = tuple(range(dim))
        self.subsets = [self.indices]
        score = self.calc_score(X_train, y_train, X_test, y_test, self.indices)

        self.scores = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices, r=dim - 1):
                score = self.calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            dim -= 1

            self.scores.append(scores[best])
        self.k_score = self.scores[-1]

        return self
    
    def transform(self, X):
        return X[:, self.indices]

    def calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(knn, k_features=1, random_state=2)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets]

plt.plot(k_feat, sbs.scores, marker='o')
plt.ylim([0, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
