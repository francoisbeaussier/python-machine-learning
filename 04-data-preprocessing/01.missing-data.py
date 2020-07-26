import pandas as pd
from io import StringIO

csv_data = \
    '''A,B,C,D
    1,2,3,4
    5,6,,8
    10,11,12,'''
df = pd.read_csv(StringIO(csv_data))

print("\nData:")
print(df)

# Find null values

print('\nCounting null values:'),
print(df.isnull().sum())

# Drop rows with null values

print('\nDrop rows with null values:')
print(df.dropna(axis=0))

# Drop columns with null values

print('\nDrop columns with null values:')
print(df.dropna(axis=1))

# dropna can also be used to remove only rows and column that are all NaN

print('\nDrop rows that are only filled with null values:')
print(df.dropna(how='all'))

# dropna can use a threshold

print('\nDrop rows that have fewer than 4 values:')
print(df.dropna(thresh=4))

# dropna can target specifc columns

print('\nDrop rows that have nulls in column C:')
print(df.dropna(subset=['C']))

# While convenient, simply removing data with null values is often not a good idea 
# because we probably also lose other valuable information

# Replace nulls by the mean value (or median or most_frequent)

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)

print('\nReplaced by mean:')
print(imputed_data)

# Pandas has a shortcut method:

print('\Pandas fillna by mean:')
print(df.fillna(df.mean()))
