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

from sklearn.model_selection import cross_val_score

# nested cross-validation 

from sklearn.model_selection import GridSearchCV
import numpy as np

param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}
]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)

scores = cross_val_score(estimator=gs, X=X_train, y=y_train, scoring='accuracy', cv=5)

print(f'CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# Compare with a tree classifier

from sklearn.tree import DecisionTreeClassifier

param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=param_grid, scoring='accuracy', cv=2)

scores = cross_val_score(estimator=gs, X=X_train, y=y_train, scoring='accuracy', cv=5)

print(f'CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
