import tensorflow as tf

print('\Batching')

a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

for item in ds:
    print(item)

ds_batch = ds.batch(3)
for i, elem in enumerate(ds_batch, 1):
    print(f'Batch {i}: {elem.numpy()}')

# combining tensors into a dataset

tf.random.set_seed(1)

print('\nCombining tensors (zip)')

t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
for example in ds_joint:
    print('x:', example[0].numpy(), '\ty:', example[1].numpy())

print('\nAlternative way')
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y)) # alternative way
for example in ds_joint:
    print('x:', example[0].numpy(), '\ty:', example[1].numpy())

ds_trans = ds_joint.map(lambda x, y: (x*2-1, y))
for example in ds_trans:
    print('x:', example[0].numpy(), '\ty:', example[1].numpy())

# Shuffle, batch and repeat

print('\nShuffle, batch and repeat')
tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=len(t_x))
for example in ds:
    print('x:', example[0].numpy(), '\ty:', example[1].numpy())

ds = ds_joint.batch(batch_size=3, drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print('Batch-x:\n', batch_x.numpy())
print('Batch-y:\n', batch_y.numpy())

ds = ds_joint.batch(batch_size=3).repeat(2)
for i, (batch_x, batch_y) in enumerate(ds):
    print('x:', batch_x.shape, '\ty:', batch_y.numpy())

print('\nshuffle(4), batch(2), repeat(3): (correct order)')
tf.random.set_seed(1)
ds = ds_joint.shuffle(4).batch(2).repeat(3)
for i, (batch_x, batch_y) in enumerate(ds):
    print('x:', batch_x.shape, '\ty:', batch_y.numpy())
