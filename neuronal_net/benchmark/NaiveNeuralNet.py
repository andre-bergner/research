import numpy as np
from itertools import accumulate
import time
import cmath
from tensorflow.examples.tutorials.mnist import input_data



def sigmoid(x):        return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):  return sigmoid(x) * (1-sigmoid(x))

def nabla_cost(v,y):
   return  v - y




class NeuralNet:

   def __init__(self, sizes, func=sigmoid, dfunc=sigmoid_deriv ):

      np.random.seed(1337)

      self.num_layers = len(sizes)
      self.sizes      = sizes
      self.biases     = [ np.random.randn(s) for s in sizes[1:] ]
      self.weights    = [ np.random.randn(m,n) for n,m in zip(sizes[:-1],sizes[1:]) ]
      self.func       = np.vectorize(func)
      self.dfunc      = np.vectorize(dfunc)

   def layer(self,w,b,x):
      u = np.dot(w,x) + b   # node input
      return  u, self.func(u)

   def __call__(self, x):
      for w,b in zip( self.weights, self.biases ):
         u,x = self.layer(w,b,x)
      return x


   # -------------------
   # learning part

   def back_propagate(self, x, y):

      # 1. feed forward an store all node input and outputs

      v = x
      layer_inputs = []
      layer_outputs = [v]
      for w,b in zip( self.weights, self.biases ):
         u,v = self.layer(w,b,v)
         layer_inputs.append(u)
         layer_outputs.append(v)

#      print("==========================")
#      for i,o in zip(layer_inputs,layer_outputs[:-1]):
#         print(o)
#         print(i)
#      print("==========================")

      # 2. back propagate error to compute all partial derivatives

      error = nabla_cost(layer_outputs[-1],y) * self.dfunc(layer_inputs[-1])
      dws = [ np.outer(error, layer_outputs[-2]) ]
      dbs = [ error ]

      for w,u,v in zip( self.weights[::-1], layer_inputs[-2::-1], layer_outputs[-3::-1] ):
         error = np.dot(w.T,error) * self.dfunc(u)
         dws.append( np.outer(error,v) )
         dbs.append( error )

      return  dws[::-1] , dbs[::-1]


   def step_gradient_descent(self, pairs, eta=0.1):

      dws = [ np.zeros(w.shape) for w in self.weights ]
      dbs = [ np.zeros(b.shape) for b in self.biases ]

      for x,y in pairs:
         dws_,dbs_ = self.back_propagate(x,y)
         for dw,dw_ in zip(dws,dws_): dw += dw_
         for db,db_ in zip(dbs,dbs_): db += db_

      for w,dw in zip(self.weights,dws): w -= dw * eta / len(pairs)
      for b,db in zip(self.biases,dbs):  b -= db * eta / len(pairs)


   def stochastic_gradient_descent(self, training_data, n_epochs=100, mini_batch_size=20, eta=0.1):

      n = len(training_data)

      for j in range(n_epochs):

         #np.random.shuffle(training_data)
         mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

         for n_batch, mini_batch in enumerate(mini_batches):
            self.step_gradient_descent( mini_batch, eta )
            print( "\rEpoch {0}/{1}, {2:.0f}%   "
                 . format(j, n_epochs, 100.*float(n_batch)/len(mini_batches))
                 , end="", flush=True)

         #print("\rEpoch {0}/{1} complete".format(j,n_epochs), end="", flush=True)

      print("")



class Timer:

   def __enter__(self):
      self.t1 = time.time()
      return self

   def __exit__(self,t,v,tb):
      t2 = time.time()
      print("{0:.4f} seconds".format(t2-self.t1))





np.set_printoptions(precision=4)



net = NeuralNet([784, 30, 10])

print("___________________________________________________")
print("Loading MNIST data set...")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
training_data = np.array([ (np.array(x),n) for x,n in zip(mnist.train.images, mnist.train.labels) ])

#training_data, validation_data, test_data = load_mnist()
#training_data = reshape_mnist_data(training_data)
#test_data = reshape_mnist_data(test_data)


print("___________________________________________________")
print("Training...")
with Timer() as t:
   net.stochastic_gradient_descent( training_data, n_epochs=2, mini_batch_size=20, eta=0.1 )


correct_results = [ np.argmax(num) == np.argmax(net(img)) for img, num in training_data ]
print("correct results: {0:.2f} %".format( float(np.count_nonzero(correct_results)) / len(correct_results) ) )


