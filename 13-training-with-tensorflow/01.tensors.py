import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)

a = np.array([1, 2 ,3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)

t_ones = tf.ones((2, 3))
print(t_ones.shape)
print(t_ones.numpy())

const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)
print(const_tensor)

t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, '--->', t_tr.shape)

t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)

t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4))
print(t.shape, '--->', t_sqz.shape)
