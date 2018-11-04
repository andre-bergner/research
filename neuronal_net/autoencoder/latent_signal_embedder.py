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
#enc = F.noise(0.01) >> F.dense([FRAME_SIZE//8]) >> act() >> F.dense([LATENT_DIM]) >> act()
enc = F.dense([FRAME_SIZE//8]) >> act() >> F.dense([LATENT_DIM]) >> act()
dec = F.dense([FRAME_SIZE//8]) >> act() >> F.dense([FRAME_SIZE])

q = L.Input([1])
z = L.concatenate([enc(x), q])
y = dec(z)
m = Model([x,q],y)


eta = 0.001
weights = [q, *m.trainable_weights]
loss = K.sum(K.square(y - x))
grad = K.gradients(loss, weights)
# grad_f = K.function(
#     m.input,
#     [*grad, loss],
#     updates=[(c, c-eta*d) for (c,d) in zip(m.trainable_weights, grad[1:])],
# )
grad_f = K.function( [x, q], [*grad, loss] )



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



def minimize_loss(input, ext_weight, eta=eta):
   *grad, loss = grad_f([input, ext_weight])
   # --- using updates -------------------------
   # ext_weight -= eta * grad[0]
   # return loss
   # --- computing all weights directly --------
   ws = [w.get_value() for w in m.trainable_weights]
   for c,d in zip([ext_weight, *ws], grad):
      c -= eta*d
   for w, w_ in zip(m.trainable_weights, ws): w.set_value(w_)
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

print(m.get_weights()[0])
losses = stochastic_gradient_descent(minimize_loss, [frames, Q], n_epochs=40)
print(m.get_weights()[0])

figure()
plot(losses)

figure()
plot(m.predict([frames[500:501], Q[500:501]])[0])
plot(frames[500])

figure()
plot(Q)



# d3v.d3viz(loss, 'loss.html')

