import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

import sys
sys.path.append('../')

from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS
from keras_tools.upsampling import UpSampling1DZeros


activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)
#act = lambda: L.LeakyReLU(alpha=0.2)  # does work too, but ugly latent space


class DenseFactory:

   def __init__(self, example_frame, latent_sizes):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])

   def make_encoder(self):
      latent_size = sum(self.latent_sizes)

      return (  F.dense([self.input_size//2])  >> act()                   # >> F.dropout(0.2)
             >> F.dense([self.input_size//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> F.dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.1)
             )

   def make_decoder(self, n):
      return (  fun._ >> XL.Slice[:, self.latent_sizes2[n]:self.latent_sizes2[n+1]]
              >> F.dense([self.input_size//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([self.input_size//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([self.input_size])
              )



class ConvFactory:

   def __init__(self, example_frame, latent_sizes, use_batch_norm=False, scale_factor=1):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.kernel_size = 5
      #self.features = [4, 8, 8, 16, 32, 32, 32]
      self.features = [4, 4, 8, 8, 16, 16, 16]
      #self.features = [2, 4, 4, 4, 8, 8, 8]
      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x
      self.scale_factor = scale_factor

      # try skip layer / residuals

   @staticmethod
   def up1d(factor=2):
      #return fun._ >> UpSampling1DZeros(factor)
      return fun._ >> L.UpSampling1D(factor)

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      return (  F.append_dimension()
             >> F.conv1d(features[0], ks, 2)  >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], ks, 2)  >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], ks, 2)  >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[5], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[6], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> F.dense([latent_size]) >> act()
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.01, no_sampling=True)
             )

   def make_decoder(self):
      up = self.up1d
      features = self.features
      ks = self.kernel_size
      return (  F.dense([self.scale_factor, features[6]]) >> act()
             >> up() >> F.conv1d(features[5], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[4], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[3], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[2], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[1], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[0], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(1, ks)
             >> F.flatten()
             )

