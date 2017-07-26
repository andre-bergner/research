import tensorflow as tf
import numpy as np
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


def stochastic_gradient_descent(net_with_loss, training_data, n_epochs=10, mini_batch_size=20, sess=None):

   if not sess:
      sess = tf.Session()
      init = tf.global_variables_initializer()
      sess.run(init)

   grad_desc = tf.train.GradientDescentOptimizer(eta)
   training_network = grad_desc.minimize(net_with_loss)

   n = len(training_data)

   losses = []

   for n_epoch in range(n_epochs):

      #np.random.shuffle(training_data)
      mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

      for n_batch, mini_batch in enumerate(mini_batches):

         avg_loss = 0.
         for input, expected in mini_batch:
            r = sess.run([training_network,net_with_loss], {net_input : input, net_expected : expected})
            avg_loss += r[1]

         losses.append(avg_loss / mini_batch_size)


         print( "\rEpoch {0}/{1}, {2:.0f}%   "
              . format(n_epoch+1, n_epochs, 100.*float(n_batch)/len(mini_batches))
              , end="", flush=True)

   print("")

   return sess, losses



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

sess, losses = stochastic_gradient_descent(net_with_loss, mnist_train, n_epochs, mini_batch_size)

# test the trained network on training data

mnist_valid = np.array([ (np.matrix(x),n) for x,n in zip(mnist.validation.images, mnist.validation.labels) ])
correct_results = [ np.argmax(num) == np.argmax(sess.run(layer2, {net_input:img})) for img, num in mnist_valid ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )


