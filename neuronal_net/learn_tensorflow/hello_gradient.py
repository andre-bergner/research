import tensorflow as tf
from pylab import *

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.Variable([1.], tf.float32)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

grad_desc = tf.train.GradientDescentOptimizer(0.01)
minimize_ax = grad_desc.minimize(0.5*(y-a*x)*(y-a*x))

def one_step():
   sess.run(minimize_ax, {x:1.,y:0.})
   return sess.run(a*x, {x:1.})[0]

plot([ one_step() for n in range(1000) ])
