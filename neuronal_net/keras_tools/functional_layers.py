import os
os.environ['KERAS_BACKEND'] = 'theano'

from functools import reduce

import keras.layers as L

import sys
sys.path.append('../')

from keras_tools import functional as fun


def dense(out_shape, use_bias=True):
   assert len(out_shape) > 0
   _ = fun.ARGS
   if len(out_shape) == 1:
      # TODO add dependent flatten
      return _ >> L.Dense(units=out_shape[0], activation=None, use_bias=use_bias)
   else:
      units = reduce(lambda x,y: x*y, out_shape)
      print("units:", units )
      return _ >> L.Dense(units=units, activation=None, use_bias=use_bias) >> L.Reshape(out_shape)


def dropout(*args, **kwargs):
   return fun.ARGS >> L.Dropout(*args, **kwargs)
