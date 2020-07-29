import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)
df_wine.columns=[
    'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoi phenols', 
    'proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# print(df_wine.head())

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Random forest

from sklearn.ensemble import RandomForestClassifier
import numpy as np

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500, random_state=2)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1] # Extented Slice -> reverse array

print('Features ranked:')
for f in range(X_train.shape[1]):
    print(f'{f+1}) {feat_labels[indices[f]]:<30} {importances[indices[f]]}')


import matplotlib.pyplot as plt

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()
plt.show()

# Note: highly correlated features may not all be ranked high

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold: {X_selected.shape[1]}')

for f in range(X_selected.shape[1]):
    print(f'{f+1}) {feat_labels[indices[f]]:<30} {importances[indices[f]]:3f}')    