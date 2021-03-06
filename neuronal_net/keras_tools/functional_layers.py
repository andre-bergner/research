from functools import reduce

import keras.layers as L

import sys
sys.path.append('../')

from keras_tools import functional as fun
from keras_tools import extra_layers as XL


def input_like(x):
   return L.Input(shape=x.shape)


def dense(out_shape, *args, **kwargs):
   assert len(out_shape) > 0
   _ = fun._
   if len(out_shape) == 1:
      # TODO add dependent flatten
      return _ >> L.Dense(units=out_shape[0], *args, **kwargs)
   else:
      units = reduce(lambda x,y: x*y, out_shape)
      return _ >> L.Dense(units=units, *args, **kwargs) >> L.Reshape(out_shape)

def conv1d(num_feat, kernel_size, stride=1, dilate=1, activation=None, use_bias=True, padding='same', *args, **kwargs):
   return fun._ >> L.Conv1D(
      num_feat,
      kernel_size,
      strides = stride,
      padding = padding,
      activation = activation,
      use_bias = use_bias,
      dilation_rate = dilate,
      *args,
      **kwargs
   )

def conv2d(num_feat, kernel_size, stride=1, activation=None, use_bias=True, padding='same', *args, **kwargs):
   return fun._ >> L.Conv2D(
      num_feat,
      kernel_size,
      strides = stride,
      padding = padding,
      activation = activation,
      use_bias = use_bias,
      *args,
      **kwargs
   )

def up1d(factor=2):
   return fun._ >> L.UpSampling1D(factor)

def up2d(factor=(2,2)):
   return fun._ >> L.UpSampling2D(factor)

def up(factor=2):
   if type(factor) == int:
      return up1d(factor)
   elif type(factor) in (list, tuple):
      if len(factor) == 1:
         return up1d(factor)
      elif len(factor) == 1:
         return up2d(factor)

def pool1d(pool_size=2, strides=None, padding='same'):
   return fun._ >> L.MaxPool1D(pool_size=pool_size, strides=strides, padding=padding)

def pool2d(pool_size=(2,2), strides=None, padding='same'):
   return fun._ >> L.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding)

def reshape(out_shape):
   return fun._ >> L.Reshape(out_shape)

def flatten(*args, **kwargs):
   return fun._ >> L.Flatten(*args, **kwargs)

def append_dimension(*args, **kwargs):
   return fun._ >> XL.AppendDimension(*args, **kwargs)

def crop1d(*args, **kwargs):
   return fun._ >> L.Cropping1D(*args, **kwargs)

def dropout(*args, **kwargs):
   return fun._ >> L.Dropout(*args, **kwargs)

def batch_norm(*args, **kwargs):
   return fun._ >> L.BatchNormalization(*args, **kwargs)

def noise(*args, **kwargs):
   return fun._ >> XL.DecayingGaussianNoise(*args, **kwargs)

def input_like(x):
   return L.Input(shape=x.shape)

def noisy_input_like(x, stddev):
   return XL.DecayingGaussianNoise(stddev, input_shape=x.shape)

