import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df = pd.read_csv(url, header=None)

from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))
pipe_lr.fit(X_train, y_train)

import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f'Fold: {k+1} Class distribution.: {np.bincount(y_train[train])}, Acc: {score:.3f}')

print(f'CV accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

# Automated cross validation

from sklearn.model_selection import cross_val_score

score = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=2)

print(f'cross_val_score accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
