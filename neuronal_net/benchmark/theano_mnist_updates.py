import theano as th
import theano.tensor as T
import numpy as np
import timer
import sgd
from pylab import *

n_epochs        = 3
mini_batch_size = 20
eta             = th.shared(value=np.float32(0.1))


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

net_input = T.fmatrix()
net_expected = T.fmatrix()

l0 = net_input
W1 = th.shared(farray(np.random.randn(784,30)))
b1 = th.shared(farray(np.random.randn(30,)))
W2 = th.shared(farray(np.random.randn(30,10)))
b2 = th.shared(farray(np.random.randn(10,)))
l1 = T.nnet.sigmoid(T.dot(l0, W1) + b1)
l2 = T.nnet.sigmoid(T.dot(l1, W2) + b2)
net_output = l2

coeffs = [W1, b1, W2, b2]

net_with_loss = ((net_expected - net_output)**2).sum()
#net_with_loss = -(net_expected*th.tensor.log(net_output) + (1.-net_expected)*th.tensor.log(1.-net_output)).sum()
net_with_loss_grad = th.grad(net_with_loss, coeffs)

net_f = th.function([net_input], net_output)
net_with_loss_grad_f = th.function(
   [net_input, net_expected],
   [*net_with_loss_grad, net_with_loss],
   updates=[(c, c-eta*d) for (c,d) in zip(coeffs, net_with_loss_grad)]
);


# load data

training_data, mnist_valid, test_data = load_mnist()
training_data = reshape_mnist_data(training_data)
mnist_valid = reshape_mnist_data(mnist_valid)
test_data = reshape_mnist_data(test_data)

def minimize_loss(input, expected, eta=eta):
   *grad, loss = net_with_loss_grad_f(input, expected)
   return loss

with timer.Timer() as t:
   losses = sgd.stochastic_gradient_descent2(minimize_loss, training_data, n_epochs, mini_batch_size)

correct_results = [
   np.argmax(num) == np.argmax(net_f(np.expand_dims(img, axis=0)))
   for img, num in zip(mnist_valid[0], mnist_valid[1])
]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

plot(losses)

