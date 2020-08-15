#%%
import tensorflow as tf
import numpy as np

a = tf.Variable(initial_value=3.14, name='var_a')
print(a)

b = tf.Variable(initial_value=[1, 2, 3], name='var_b')
print(b)

c = tf.Variable(initial_value=[True, False], name='var_c')
print(c)

d = tf.Variable(initial_value=['abc'], name='var_d')
print(d)

# %%
w = tf.Variable([1, 2, 3], trainable=False)
print('trainable is', w.trainable)

print(w.assign([3, 1, 4], read_value=True))
w.assign_add([2, -1, 2], read_value=False)
print(w.value())

#%%
tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()

tf.print(init(shape=(3,)))

v = tf.Variable(init(shape=(2, 3)))
tf.print(v)

#%%
class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)

m = MyModule()
print('All module variables: ', [v.shape for v in m.trainable_variables])
