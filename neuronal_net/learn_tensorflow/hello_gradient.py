import tensorflow as tf
from pylab import *
import timer

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.Variable([1.], tf.float32)
l = a*x

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

dist_inout = 0.5*(y-l)**2
step = tf.train.GradientDescentOptimizer(0.01).minimize(dist_inout)

def one_step():
   sess.run(step, {x:1.,y:0.})
   return sess.run(l, {x:1.})[0]

with timer.Timer() as t:
   result = [ one_step() for n in range(1000) ]

plot(result)
