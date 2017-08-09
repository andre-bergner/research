# TODO
# * plot avg gradient per layer → analyze deeper fully connected network
# * learn convolution matrix from data:
#   * input: gaussian at position x    output: dirac at position x
#   * input: rect at position x,y    output: dirac at position x,y
#   * input: rect or circle at diff postion    output  [1,0] or [0,1]   →  should learn translation
#   * train cube and hypercube at diff postions → analyze avg. node degree → emerge 6 vs 8 connections?
# * use conv-layer
# * cross-corr 
# * regularization term
# * max-pooling

import theano as th
import numpy as np
import timer
import sgd
from pylab import *

n_epochs        = 3
mini_batch_size = 20
eta             = 0.1


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
#net_with_loss = -(net_expected*th.tensor.log(net_output) + (1.-net_expected)*th.tensor.log(1.-net_output)).sum()
net_with_loss_grad = th.grad(net_with_loss, [W1,b1,W2,b2])

net_f = th.function([net_input, W1, b1, W2, b2], net_output)
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

def minimize_loss(input, expected, eta=eta):
   *grad, loss = net_with_loss_grad_f(input, expected, *coeffs)
   for c,d in zip(coeffs, grad):
      c -= eta*d
   return loss

with timer.Timer() as t:
   losses = sgd.stochastic_gradient_descent(minimize_loss, training_data, n_epochs, mini_batch_size)

correct_results = [ np.argmax(num) == np.argmax(net_f(img,*coeffs)) for img, num in mnist_valid ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

plot(losses)

