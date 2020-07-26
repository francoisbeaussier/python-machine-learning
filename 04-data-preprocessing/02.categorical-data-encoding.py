import pandas as pd

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])

df.columns = ['color', 'size', 'price', 'classlabel']

print('dataset:')
print(df)

# Mapping ordinal features

size_mappings = {'XL': 4, 'L': 3, 'M': 2, 'S': 1}

df['size'] = df['size'].map(size_mappings)

inv_size_mapping = { v: k for k, v in size_mappings.items() }

print('\nMapping back:')
print(df['size'].map(inv_size_mapping))

# Encoding class labels 

import numpy as np

class_mappings = { label: idx for idx, label in enumerate(df['classlabel'].unique()) }
print('\nClass labels:')
print(class_mappings)

df['classlabel'] = df['classlabel'].map(class_mappings)

inv_class_mapping = { v: k for k, v in class_mappings.items() }

print('\nMapping back:')
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df['classlabel'])

# Using sklearn

from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

print('\nsklearn class labels:')
print(y)

print('\nMapping back:')
print(class_le.inverse_transform(y))

# One hot encoding

from sklearn.preprocessing import OneHotEncoder

X = df[['color', 'size', 'price']].values
print(X)

color_ohe = OneHotEncoder()
ohe = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

print('\One hot encoded:')
print(ohe)

from sklearn.compose import ColumnTransformer

c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]), # can also use categories='auto' and drop='first' to drop one column
    ('nothing', 'passthrough', [1, 2])
])

print('Recompose the whole table:')
print(c_transf.fit_transform(X).astype(float))

# Pandas has a built-in helper as well

print("Recompose with pandas get_dummies:")
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))
