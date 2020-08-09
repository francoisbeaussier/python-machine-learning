from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    def fit(self, X, y):
        self.labenc = LabelEncoder()
        self.labenc.fit(y)
        self.classes = self.labenc.classes_
        self.classifiers = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labenc.transform(y))
            self.classifiers.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(k, weights=self.weights)), axis=1, arr=predictions)
        maj_vote = self.labenc.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    # def decision_function(self, X, *args, **kwargs):
    #     probas = np.asarray([clf.decision_function(X) for clf in self.classifiers])
    #     print(probas.shape)
    #     avg_proba = np.average(probas, axis=0, weights=self.weights)
    #     print(avg_proba.shape)
    #     return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

all_clf = [pipe1, clf2, pipe3, mv_clf]
clf_labels = ['Logistic regression', 'Decision tree', 'KNN', 'Majority voting']

print('10-fold cross validation:\n')
for clf, label in zip(all_clf, clf_labels):
    print('label', y_test.shape)
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} +/- {scores.std()}')
