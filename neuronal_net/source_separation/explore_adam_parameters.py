import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.utils import plot_model

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS
from keras_tools.upsampling import UpSampling1DZeros

from timeshift_autoencoder import predictors as P
from result_tools import *
from test_data import *


frame_size = 128
n_pairs = 10000
latent_sizes = [4, 4]
n_epochs = 10
noise_stddev = 0.0

activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)


class ConvFactory:

   def __init__(self, example_frame, latent_sizes):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])
      self.kernel_size = 3
      self.features = [4, 4, 8, 8, 16, 16, 16]

   @staticmethod
   def up1d(factor=2):
      return fun._ >> L.UpSampling1D(factor)

   def make_encoder(self):
      latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      return (  F.append_dimension()
             >> F.conv1d(features[0], ks, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], ks, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], ks, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[5], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[6], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> F.dense([latent_size]) >> act()
             )

   def make_decoder(self, n):
      up = self.up1d
      features = self.features
      ks = self.kernel_size
      return (  fun._ >> XL.Slice[:, self.latent_sizes2[n]:self.latent_sizes2[n+1]]
              >> F.dense([1, features[6]]) >> act()
              >> up() >> F.conv1d(features[5], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[4], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[3], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[2], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[1], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[0], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(1, ks)
              >> F.flatten()
              )



def make_factor_model(example_frame, factory):

   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   encoder = factory.make_encoder()
   ex = (eta() >> encoder)(x)
   decoders = [factory.make_decoder(n) for n in range(len(factory.latent_sizes))]
   channels = [dec(ex) for dec in decoders]
   y = L.add(channels)

   m = M.Model([x], [y])

   return (
      m,
      M.Model([x], [encoder(x)]),
      None,
      [M.Model([x], [c]) for c in channels]
   )



class LossRecorder(keras.callbacks.Callback):

   def __init__(self, **kargs):
      super(LossRecorder, self).__init__(**kargs)
      self.losses = []
      self.grads = []
      self.pred_errors = []

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights

   def on_epoch_end(self, epoch, logs={}):
      self.pred_errors.append(
      [   pred_error(mode1, frames, sig1, 2048)
      ,   pred_error(mode1, frames, sig2, 2048)
      ,   pred_error(mode2, frames, sig2, 2048)
      ,   pred_error(mode2, frames, sig1, 2048)
      ])


sig1, sig2 = lorenz_fm
sig_gen = sig1 + sig2

frames = np.array([w for w in windowed(sig_gen(n_pairs+frame_size-1), frame_size, 1)])

factory = ConvFactory
model, encoder, model_sf, [mode1, mode2] = make_factor_model(frames[0], factory(frames[0], latent_sizes))

model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.8, beta_2=0.999, decay=0.0001), loss='mse')
model.load_weights('init_weights_1.hdf5')

loss_recorder = LossRecorder()

def train_epoch():

   batch_size = 128
   #stride = 10
   #batches = np.array([frames[n:n+batch_size*stride:stride] for n in np.arange(n_pairs-batch_size*stride) ])

   batches = np.array([w for w in windowed(frames, batch_size)])

   for b in batches:
      loss = model.train_on_batch(b,b)
      loss_recorder.losses.append(loss)

   loss_recorder.pred_errors.append(
   [   pred_error(mode1, frames, sig1, 2048)
   ,   pred_error(mode1, frames, sig2, 2048)
   ,   pred_error(mode2, frames, sig2, 2048)
   ,   pred_error(mode2, frames, sig1, 2048)
   ])

   print(loss_recorder.losses[-1], loss_recorder.pred_errors[-1])

for n in range(200):
   train_epoch()

training_summary(model, mode1, mode2, encoder, sig_gen, sig1, sig2, frames, loss_recorder)

