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


def make_conv1d(num_feat, kernel_size, stride, activation, initializer, use_bias, padding='same'):
   return L.Conv1D(
      num_feat,
      kernel_size,
      padding = padding,
      strides = stride,
      activation = activation,
      kernel_initializer = initializer,
      use_bias = use_bias,
   )

def make_conv2d(num_feat, kernel_size, stride, activation, initializer, use_bias, padding='same'):
   return L.Conv2D(
      num_feat,
      (kernel_size, kernel_size),
      padding = padding,
      strides = (stride, stride),
      activation = activation,
      kernel_initializer = initializer,
      use_bias = use_bias,
   )


convolver = {
   '1d': make_conv1d,
   '2d': make_conv2d
}


upsampler = {
   '1d': lambda factor: L.UpSampling1D(factor),
   '2d': lambda factor: L.UpSampling2D((factor, factor))
}



def make_autoencoder(size, n_features, activation='tanh', use_bias=True, init=keras.initializers.VarianceScaling()):

   if type(size) == int:
      size = [size]

   dim = str(len(size)) + 'd'

   act = lambda: L.Activation(activation)
   up = upsampler[dim]
   conv = lambda n_feat, stride: convolver[dim](n_feat, 5, stride, activation, init, use_bias)
   _ = fun.ARGS

   x = L.Input(shape=size)
   reshape_in = _ >> L.Reshape(size + [1])

   enc1 = _ >> conv(n_features,   2) >> act()
   enc2 = _ >> conv(n_features*2, 2) >> act()
   enc3 = _ >> conv(n_features*4, 2) >> act()

   enc_core = _ >> convolver[dim](n_features, 2, 2, activation, init, use_bias, padding='valid') >> act()

   dec1 = _ >> up(4) >> conv(n_features*2, 1) >> act()
   dec2 = _ >> up(2) >> conv(n_features, 1) >> act()
   dec3 = _ >> up(2) >> conv(1, 1) >> act()

   reshape_out = L.Reshape(size)

   y1 = reshape_in >> enc1 >> dec3 >> reshape_out
   y2 = reshape_in >> enc1 >> enc2 >> dec2 >> dec3 >> reshape_out
   y3 = reshape_in >> enc1 >> enc2 >> enc3 >> enc_core >> dec1 >> dec2 >> dec3 >> reshape_out

   model = M.Model([x], [y3(x)])
   joint_model = M.Model([x], [y1(x),y2(x),y3(x)])

   return model, joint_model


