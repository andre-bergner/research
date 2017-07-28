import tensorflow as tf
import numpy as np
import timer
import sgd
from pylab import *

from tensorflow.examples.tutorials.mnist import input_data

n_epochs        = 3
mini_batch_size = 20
eta             = 0.1


def make_dense_layer( num_inputs, num_outputs, x ):

   W = tf.Variable(tf.random_normal([num_inputs,num_outputs]), tf.float32)
   b = tf.Variable(tf.random_normal([num_outputs]), tf.float32)

   return  tf.sigmoid( tf.matmul(x,W) + b )


def make_loss( layer, expected ):

   return tf.reduce_sum(tf.square(layer - expected))


# build network

net_input = tf.placeholder(tf.float32, shape=[None,784])
net_expected = tf.placeholder(tf.float32, shape=[10])

layer1 = make_dense_layer( 784, 30, net_input )
layer2 = make_dense_layer( 30, 10, layer1 )
net_with_loss = make_loss( layer2, net_expected )

# load data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_train = np.array([ (np.matrix(x),n) for x,n in zip(mnist.train.images, mnist.train.labels) ])

# train

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

grad_desc = tf.train.GradientDescentOptimizer(eta)
training_network = grad_desc.minimize(net_with_loss)

def minimize_loss(input, expected, eta=eta):
   _,loss = sess.run([training_network,net_with_loss], {net_input : input, net_expected : expected})
   return loss


with timer.Timer() as t:
   losses = sgd.stochastic_gradient_descent(minimize_loss, mnist_train, n_epochs, mini_batch_size)

# test the trained network on training data

mnist_valid = np.array([ (np.matrix(x),n) for x,n in zip(mnist.validation.images, mnist.validation.labels) ])
correct_results = [ np.argmax(num) == np.argmax(sess.run(layer2, {net_input:img})) for img, num in mnist_valid ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

plot(losses)

