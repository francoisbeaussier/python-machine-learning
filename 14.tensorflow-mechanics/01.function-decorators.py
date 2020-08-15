import tensorflow as tf
import numpy as np

@tf.function
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(r1, 2)
    z = tf.add(r2, c)
    return z

tf.print('Scalar inputs:', compute_z(1, 2, 3))
tf.print('Rank 1 inputs:', compute_z([1], [2], [3]))
tf.print('Rank 2 inputs:', compute_z([[1]], [[2]], [[3]]))

# Note, the first parameter of the shape is None meaning it's flexible as it will depend on the batch size!

@tf.function(input_signature=(
                            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),))
def compute_z2(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(r1, 2)
    z = tf.add(r2, c)
    return z

# tf.print('Scalar inputs:', compute_z2(1, 2, 3))
# tf.print('Rank 1 inputs:', compute_z2([1], [2], [3]))  
tf.print('Rank 2 inputs:', compute_z2([[1]], [[2]], [[3]]))