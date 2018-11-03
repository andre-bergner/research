import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
import keras.optimizers as O

from keras.utils import plot_model

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS

from pylab import *

from keras.models import Model
from keras.utils.generic_utils import unpack_singleton
import theano.d3viz as d3v


# ---------------------------------------------------------------------------------------
# MODEL

FRAME_SIZE = 128
LATENT_DIM = 1

activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)

x = L.Input([FRAME_SIZE])
# enc = F.dense([FRAME_SIZE//4]) >> act() >> F.dense([FRAME_SIZE//8]) >> act() >> F.dense([LATENT_DIM]) >> act()
# dec = F.dense([FRAME_SIZE//8]) >> act() >> F.dense([FRAME_SIZE//4]) >> act() >> F.dense([FRAME_SIZE])
enc = F.noise(0.1) >> F.dense([FRAME_SIZE//8]) >> act() >> F.dense([LATENT_DIM]) >> act()
dec = F.dense([FRAME_SIZE//8]) >> act() >> F.dense([FRAME_SIZE])

q = L.Input([1])
z = L.concatenate([enc(x), q])
y = dec(z)
m = Model([x,q],y)



# ---------------------------------------------------------------------------------------
# TRAINING DATA

def windowed(xs, win_size, hop=None):
   if hop == None: hop = win_size
   if win_size <= len(xs):
      for n in range(0, len(xs)-win_size+1, hop):
         yield xs[n:n+win_size]

rng = np.arange(5000)
X = np.sin(0.1*rng) * 0.5 * (1 + np.cos(0.00212*rng))

frames = np.array([w for w in windowed(X, FRAME_SIZE, 1)])
#Q = np.random.randn(len(frames),1)  # the latent code that should be learned
Q = np.zeros([len(frames),1])  # the latent code that should be learned


m.compile('sgd', 'mse')

eta = 0.001
weights = m.trainable_weights
loss = K.sum(K.square(m.output - m.input[0]))
grad = K.gradients(loss, [m.input[1], *weights])
grad_f = K.function(
    m.input,
    [*grad, loss],
    updates=[(c, c-eta*d) for (c,d) in zip(weights, grad[1:])]
)

def minimize_loss(input, ext_weight, eta=eta):
   *grad, loss = grad_f([input, ext_weight])
   ext_weight -= eta * grad[0]
   return loss

def stochastic_gradient_descent(minimizer, training_data, n_epochs=3, mini_batch_size=32):

   n_data = len(training_data[0])
   n_batches = n_data//mini_batch_size

   losses = []

   for n_epoch in range(n_epochs):

      for n_batch in range(n_batches):

         idx = np.random.randint(0, n_data, mini_batch_size)
         input_batch = training_data[0][idx]
         exwgt_batch = training_data[1][idx]

         loss = minimizer(input_batch, exwgt_batch)

         training_data[1][idx] = exwgt_batch    # copy back updated values
         
         losses.append(loss / mini_batch_size)

         print( "\rEpoch {0}/{1}, {2:.0f}%   "
              . format(n_epoch+1, n_epochs, 100.*float(n_batch)/n_batches)
              , end="", flush=True)
   print("")

   return losses


losses = stochastic_gradient_descent(minimize_loss, [frames, Q], n_epochs=40)

figure()
plot(losses)

figure()
plot(m.predict([frames[500:501], Q[500:501]])[0])
plot(frames[500])

figure()
plot(Q)



# d3v.d3viz(loss, 'loss.html')

