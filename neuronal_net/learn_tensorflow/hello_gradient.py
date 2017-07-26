import tensorflow as tf
from pylab import *
import timer

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.Variable([1.], tf.float32)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

dist_inout = 0.5*(y-a*x)*(y-a*x)
step = tf.train.GradientDescentOptimizer(0.01).minimize(dist_inout)

def one_step():
   sess.run(step, {x:1.,y:0.})
   return sess.run(a*x, {x:1.})[0]

with timer.Timer() as t:
   result = [ one_step() for n in range(1000) ]

plot(result)
