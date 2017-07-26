# TODO
# * plot avg gradient per layer → analyze deeper fully connected network
# * use conv-layer
# * cross-corr 
# * regularization term
# * max-pooling

import theano as th
import numpy as np
import timer
from pylab import *

n_epochs        = 3
mini_batch_size = 20
eta             = 0.1


def stochastic_gradient_descent(minimizer, training_data, n_epochs=10, mini_batch_size=20):

   n = len(training_data)

   losses = []

   for n_epoch in range(n_epochs):

      np.random.shuffle(training_data)
      mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

      for n_batch, mini_batch in enumerate(mini_batches):

         avg_loss = 0.
         for input, expected in mini_batch:
            loss = minimizer(input, expected)
            avg_loss += loss

         losses.append(avg_loss / mini_batch_size)

         print( "\rEpoch {0}/{1}, {2:.0f}%   "
              . format(n_epoch+1, n_epochs, 100.*float(n_batch)/len(mini_batches))
              , end="", flush=True)

   print("")

   return losses


def farray(a):
   return np.array(a,dtype=np.float32)


def load_mnist( path = "mnist.pkl.gz" ):

   import gzip
   import pickle

   with gzip.open(path, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
   
   return data  # training_data, validation_data, test_data

def reshape_mnist_data( data ):
   def dirac(n):
      d = np.zeros(10); d[n]=1; return d;
   return [ (farray(x),farray(dirac(n))) for x,n in zip(data[0],data[1]) ]


# build network

net_input = th.tensor.fvector()
net_expected = th.tensor.fvector()

l0 = net_input
W1 = th.tensor.fmatrix()
b1 = th.tensor.fvector()
l1 = th.tensor.nnet.sigmoid(W1.dot(l0) + b1)
W2 = th.tensor.fmatrix()
b2 = th.tensor.fvector()
l2 = th.tensor.nnet.sigmoid(W2.dot(l1) + b2)
net_output = l2

net_with_loss = ((net_expected - net_output)**2).sum()
net_with_loss_grad = th.grad(net_with_loss, [W1,b1,W2,b2])

net_f = th.function([net_input, W1, b1, W2, b2], net_output)
#net_with_loss_f = th.function([net_input, net_expected, W1, b1, W2, b2], net_with_loss)
net_with_loss_grad_f = th.function( [net_input, net_expected, W1, b1, W2, b2]
                                  , [*net_with_loss_grad, net_with_loss] );


cW1 = farray(np.random.randn(30,784))
cb1 = farray(np.random.randn(30))
cW2 = farray(np.random.randn(10,30))
cb2 = farray(np.random.randn(10))

# load data

training_data, mnist_valid, test_data = load_mnist()
training_data = reshape_mnist_data(training_data)
mnist_valid = reshape_mnist_data(mnist_valid)
test_data = reshape_mnist_data(test_data)

coeffs = [cW1, cb1, cW2, cb2]

def minimize_loss(input, expected, eta=.5):
   *grad, loss = net_with_loss_grad_f(input, expected, *coeffs)
   for c,d in zip(coeffs, grad):
      c -= eta*d
   return loss

with timer.Timer() as t:
   losses = stochastic_gradient_descent(minimize_loss, training_data, n_epochs, mini_batch_size)

correct_results = [ np.argmax(num) == np.argmax(net_f(img,*coeffs)) for img, num in mnist_valid ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

plot(losses)

