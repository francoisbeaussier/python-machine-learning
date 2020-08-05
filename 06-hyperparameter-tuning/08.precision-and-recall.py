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

pipe_svc.fit(X_train, y_train)

y_pred = pipe_svc.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print(f'Precision   : {precision_score(y_true=y_test, y_pred=y_pred):.3f}')
print(f'Recall      : {recall_score(y_true=y_test, y_pred=y_pred):.3f}')
print(f'F1          : {f1_score(y_true=y_test, y_pred=y_pred):.3f}')

from sklearn.metrics import make_scorer

c_gamma_range = [0.01, 0.1, 1, 10]

param_grid = [
    {'svc__C': c_gamma_range, 'svc__kernel': ['linear']},
    {'svc__C': c_gamma_range, 'svc__kernel': ['rbf'], 'svc__gamma': c_gamma_range}
]

scorer = make_scorer(f1_score, pos_label=0)

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)