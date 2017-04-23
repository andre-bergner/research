import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32)

def make_layer( x ):

   W = tf.Variable([+.3], tf.float32)
   b = tf.Variable([-.3], tf.float32)

   return  tf.sigmoid( W*x + b )


layer1 = make_layer( x )
layer2 = make_layer( layer1 )

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

x0 = 0.5
y1 = sess.run(layer1, {x:x0})
y = sess.run(layer2, {x:x0})


#   -------------------------------------------------------
#  TEST
#   -------------------------------------------------------


def sigmoid(x): return 1. / (1. + np.exp(-x))
l1 = sigmoid(.3*x0 - .3)
l2 = sigmoid(.3*l1 - .3)
assert( 1e-6 > abs(y1[0] - l1) )
assert( 1e-6 > abs(y[0] - l2) )
