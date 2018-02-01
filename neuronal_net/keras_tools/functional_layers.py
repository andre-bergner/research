import os
os.environ['KERAS_BACKEND'] = 'theano'

from functools import reduce

import keras.layers as L

import sys
sys.path.append('../')

from keras_tools import functional as fun


def dense(out_shape, activation=None, use_bias=True):
   assert len(out_shape) > 0
   _ = fun._
   if len(out_shape) == 1:
      # TODO add dependent flatten
      return _ >> L.Dense(units=out_shape[0], activation=activation, use_bias=use_bias)
   else:
      units = reduce(lambda x,y: x*y, out_shape)
      print("units:", units )
      return _ >> L.Dense(units=units, activation=activation, use_bias=use_bias) >> L.Reshape(out_shape)


def conv1d(num_feat, kernel_size, stride, activation = None, use_bias=True, padding='same', *args, **kwargs):
   return fun._ >> L.Conv1D(
      num_feat,
      kernel_size,
      strides = stride,
      padding = padding,
      activation = activation,
      use_bias = use_bias,
      *args,
      **kwargs
   )


def flatten(*args, **kwargs):
   return fun._ >> L.Flatten(*args, **kwargs)

def dropout(*args, **kwargs):
   return fun._ >> L.Dropout(*args, **kwargs)

def batch_norm(*args, **kwargs):
   return fun._ >> L.BatchNormalization(*args, **kwargs)

def noise(stddev, *args, **kwargs):
   return fun._ >> L.GaussianNoise(stddev)

def input_like(x):
   return L.Input(shape=x.shape)

def noisy_input_like(x, stddev):
   return L.GaussianNoise(stddev, input_shape=x.shape)
