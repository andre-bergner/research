import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import scipy.signal as ss

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import upsampling as Up
from keras_tools import functional as fun


def make_conv1d(num_feat, kernel_size, stride, initializer, use_bias, padding='same'):
   return L.Conv1D(
      num_feat,
      kernel_size,
      padding = padding,
      strides = stride,
      activation = None,
      kernel_initializer = initializer,
      use_bias = use_bias,
   )

def make_conv2d(num_feat, kernel_size, stride, initializer, use_bias, padding='same'):
   return L.Conv2D(
      num_feat,
      (kernel_size, kernel_size),
      padding = padding,
      strides = (stride, stride),
      activation = None,
      kernel_initializer = initializer,
      use_bias = use_bias,
   )


def input_like(x):
   return L.Input(shape=x.shape)


convolver = {
   '1d': make_conv1d,
   '2d': make_conv2d
}


upsampler = {
   '1d': lambda factor: L.UpSampling1D(factor),
   '2d': lambda factor: L.UpSampling2D((factor, factor))
}


def make_model(size, model_generator, activation='tanh', use_bias=True, init=keras.initializers.VarianceScaling()):

   if type(size) == int:
      size = [size]

   dim = str(len(size)) + 'd'

   _ = fun.ARGS
   act = lambda: L.Activation(activation)    # TODO make activation type a parameter
   up = lambda f: _ >> upsampler[dim](f)
   # conv = lambda n_feat, stride: _ >> convolver[dim](n_feat, 5, stride, activation, init, use_bias)
   def conv(n_feat, kernel_size, stride=1, initializer=init, use_bias=use_bias, padding='same'):
      return _ >> convolver[dim](n_feat, kernel_size, stride, initializer, use_bias, padding)

   x = L.Input(shape=size)
   reshape_in = _ >> L.Reshape(size + [1])
   reshape_out = L.Reshape(size)

   return model_generator(x, reshape_in, reshape_out, conv, act, up)



def make_autoencoder(size, n_features, activation='tanh', use_bias=True, init=keras.initializers.VarianceScaling()):

   def model_gen(x, reshape_in, reshape_out, conv, act, up):

      enc1 = conv(n_features,   5, 2) >> act()
      enc2 = conv(n_features*2, 5, 2) >> act()
      enc3 = conv(n_features*4, 5, 2) >> act()

      enc_core = conv(n_features, 2, 2, padding='valid') >> act()

      dec1 = up(4) >> conv(n_features*2, 5) >> act()
      dec2 = up(2) >> conv(n_features, 5) >> act()
      dec3 = up(2) >> conv(1, 5) >> act()

      y1 = reshape_in >> enc1 >> dec3 >> reshape_out
      y2 = reshape_in >> enc1 >> enc2 >> dec2 >> dec3 >> reshape_out
      y3 = reshape_in >> enc1 >> enc2 >> enc3 >> enc_core >> dec1 >> dec2 >> dec3 >> reshape_out

      model = M.Model([x], [y3(x)])
      joint_model = M.Model([x], [y1(x),y2(x),y3(x)])

      return model, joint_model

   return make_model(size, model_gen, activation, use_bias, init)
