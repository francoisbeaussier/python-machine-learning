
#%%
import pandas as pd
import tensorflow as tf

dataset_path = tf.keras.utils.get_file(
    'auto-mpg.data',
    'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'ModelYear', 'Origin']

df = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

df = df.dropna()
df = df.reset_index(drop=True)

#%%

import sklearn
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.8)
train_stats = df_train.describe().transpose()

numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std
    
df_train_norm.tail()

numeric_features = []
for col_name in numeric_column_names:
    numeric_features.append(tf.feature_column.numeric_column(key=col_name))

feature_year = tf.feature_column.numeric_column(key='ModelYear')
bucketized_features = [
    tf.feature_column.bucketized_column(
        source_column=feature_year, 
        boundaries=[73, 76, 79])
]

feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Origin',
        vocabulary_list=[1, 2, 3])
categorical_indicator_features = [
    tf.feature_column.indicator_column(feature_origin)
]

def train_input_fn(df_train, batch_size=8):
    df = df_train.copy()
    train_x, train_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    return dataset.shuffle(1000).repeat().batch(batch_size)

ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
print('Keys:', batch[0].keys())
print('Batch Model Years:', batch[0]['ModelYear'])

def eval_input_fn(df_test, batch_size=8):
    df = df_test.copy()
    test_x, test_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
    return dataset.batch(batch_size)

all_feature_columns = (numeric_features + bucketized_features + categorical_indicator_features)

regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10],
    model_dir='models/autompg-dnnregressor/'
)

# %%
import numpy as np
EPOCHS = 10
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))
print('Training steps:', total_steps)

regressor.train(input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE), steps=total_steps)

#%%
import tensorflow as tf
reloaded_regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units = [32, 10],
    warm_start_from='models/autompg-dnnregressor/',
    model_dir='models/autompg-dnnregressor/'
)

#%%
eval_results = reloaded_regressor.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))
print(f'Average loss: {eval_results["average_loss"]:.4f}')

pred_res = regressor.predict(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))
print(next(iter(pred_res)))

# %%
# Let's try with Tree + Boosting

boosted_tree = tf.estimator.BoostedTreesRegressor(
    feature_columns=all_feature_columns,
    n_batches_per_layer=20,
    n_trees=200
)

boosted_tree.train(input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE))

#%%
eval_results = boosted_tree.evaluate(input_fn=lambda:eval_input_fn(df_test_norm, batch_size=BATCH_SIZE))

print(f'Average loss: {eval_results["average_loss"]:.4f}')

# We get a lower loss, which can be expected with such a small dataset
