import keras
import keras.models as M
import keras.backend as K

import numpy as np
import timer
import sgd
from pylab import *

n_epochs        = 3
mini_batch_size = 20
eta             = np.float32(0.1)


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
   return data[0], farray([farray(dirac(n)) for n in data[1]])


# build network

net_input = K.placeholder(ndim=2)
net_expected = K.placeholder(ndim=2)

l0 = net_input
W1 = K.placeholder(ndim=2)
b1 = K.placeholder(ndim=1)
W2 = K.placeholder(ndim=2)
b2 = K.placeholder(ndim=1)
l1 = K.sigmoid(K.dot(l0, W1) + b1)
l2 = K.sigmoid(K.dot(l1, W2) + b2)
net_output = l2

coeffs = [W1, b1, W2, b2]

net_with_loss = K.sum((net_expected - net_output)**2)
#net_with_loss = -(net_expected*T.log(net_output) + (1.-net_expected)*T.log(1.-net_output)).sum()
net_with_loss_grad = K.gradients(net_with_loss, coeffs)

net_f = K.function([net_input, W1, b1, W2, b2], [net_output])
net_with_loss_grad_f = K.function(
   [net_input, net_expected, *coeffs],
   [*net_with_loss_grad, net_with_loss]
);


cW1 = farray(np.random.randn(784,30))
cb1 = farray(np.random.randn(30,))
cW2 = farray(np.random.randn(30,10))
cb2 = farray(np.random.randn(10,))

# load data

training_data, mnist_valid, test_data = load_mnist()
training_data = reshape_mnist_data(training_data)
mnist_valid = reshape_mnist_data(mnist_valid)
test_data = reshape_mnist_data(test_data)

vcoeffs = [cW1, cb1, cW2, cb2]

def minimize_loss(input, expected, eta=eta):
   *grad, loss = net_with_loss_grad_f([input, expected, *vcoeffs])
   for c,d in zip(vcoeffs, grad):
      c -= eta*d
   return loss

with timer.Timer() as t:
   losses = sgd.stochastic_gradient_descent2(minimize_loss, training_data, n_epochs, mini_batch_size)

correct_results = [
   np.argmax(num) == np.argmax(net_f([np.expand_dims(img, axis=0), *vcoeffs]))
   for img, num in zip(mnist_valid[0], mnist_valid[1])
]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

plot(losses)
