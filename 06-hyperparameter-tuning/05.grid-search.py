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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pipe_svc = make_pipeline(StandardScaler(),
                        SVC(random_state=1))

from sklearn.model_selection import GridSearchCV

import numpy as np

param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}
]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('Best score:', gs.best_score_)
print('Best params:', gs.best_params_)

clf = gs.best_estimator_

print('Test accuracy:', clf.score(X_test, y_test))

# Note: using RandomizedSearch might perform just as well with much better performance