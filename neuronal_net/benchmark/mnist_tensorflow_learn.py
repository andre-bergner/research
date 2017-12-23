import tensorflow as tf
import numpy as np
import time
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


def model(features, labels, mode):

   # build network

   #net_input = tf.placeholder(tf.float32, shape=[None,784])
   #net_expected = tf.placeholder(tf.float32, shape=[10])

   net_input = features["net_input"]
   layer1 = make_dense_layer( 784, 30, net_input )
   layer2 = make_dense_layer( 30, 10, layer1 )
   y = layer2
   net_with_loss = make_loss( y, labels )

   global_step = tf.train.get_global_step()
   optimizer = tf.train.GradientDescentOptimizer(eta)
   train = tf.group(optimizer.minimize(net_with_loss), tf.assign_add(global_step, 1))

   return tf.contrib.learn.ModelFnOps( mode = mode
                                     , predictions = y
                                     , loss = net_with_loss
                                     , train_op = train )


class Timer:

   def __enter__(self):
      self.t1 = time.time()
      return self

   def __exit__(self,t,v,tb):
      t2 = time.time()
      print("{0:.4f} seconds".format(t2-self.t1))


estimator = tf.contrib.learn.Estimator(model_fn=model)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#x = np.array([ np.matrix(x) for x in mnist.train.images ])
x = np.array(mnist.train.images, dtype=np.float32)
y = np.array(mnist.train.labels, dtype=np.float32)
input_fn = tf.contrib.learn.io.numpy_input_fn({"net_input": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=1000))

predictions = [k for k in estimator.predict(input_fn=tf.contrib.learn.io.numpy_input_fn({"net_input": x}))]
correct_results = [ np.argmax(expl) == np.argmax(pred) for expl, pred in zip(y,predictions) ]
print("correct results: {0:.2f} %".format( float(np.count_nonzero(correct_results)) / len(correct_results) ) )


