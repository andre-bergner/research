import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

n_epochs        = 10
mini_batch_size = 20
eta             = 0.1


def make_dense_layer( num_inputs, num_outputs, x ):

   W = tf.Variable(tf.random_normal([num_inputs,num_outputs]), tf.float32)
   b = tf.Variable(tf.random_normal([num_outputs]), tf.float32)

   return  tf.sigmoid( tf.matmul(x,W) + b )


def make_loss( layer, expected ):

   return tf.reduce_sum(tf.square(layer - expected))


def stochastic_gradient_descent(training_network, training_data, n_epochs=10, mini_batch_size=20):

   sess = tf.Session()
   init = tf.global_variables_initializer()
   sess.run(init)

   n = len(training_data)

   for n_epoch in range(n_epochs):

      mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

      for n_batch, mini_batch in enumerate(mini_batches):

         for input, expected in mini_batch:
            sess.run(training_network, {net_input : input, net_expected : expected})

         print( "\rEpoch {0}/{1}, {2:.0f}%   "
              . format(n_epoch, n_epochs, 100.*float(n_batch)/len(mini_batches))
              , end="", flush=True)

   print("")

   return sess



# build network

net_input = tf.placeholder(tf.float32, shape=[None,784])
net_expected = tf.placeholder(tf.float32, shape=[10])

layer1 = make_dense_layer( 784, 30, net_input )
layer2 = make_dense_layer( 30, 10, layer1 )
net_with_loss = make_loss( layer2, net_expected )

# load data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_train = np.array([ (np.matrix(x),n) for x,n in zip(mnist.train.images, mnist.train.labels) ])

# setup up optimizer

grad_desc = tf.train.GradientDescentOptimizer(eta)
net_training_setup = grad_desc.minimize(net_with_loss)
sess = stochastic_gradient_descent(net_training_setup, mnist_train, n_epochs=10)

# test the trained network on training data

correct_results = [ np.argmax(num) == np.argmax(sess.run(layer2, {net_input:img})) for img, num in mnist_train ]
print("correct results: {0:.2f} %".format( float(np.count_nonzero(correct_results)) / len(correct_results) ) )


