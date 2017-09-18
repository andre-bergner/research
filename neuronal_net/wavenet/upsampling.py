import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.layers as L
import keras.backend as K
import keras.utils.conv_utils as conv_utils

from keras.engine.topology import Layer
from keras.engine import InputSpec


from theano import tensor as TT

def up_sampling_1d_theano(x, factor):
    input_shape = x.shape
    output_shape = (input_shape[0], factor*input_shape[1], input_shape[2])
    output = TT.zeros(output_shape)
    result = TT.set_subtensor(output[:, ::factor, :], x)

    if hasattr(x, '_keras_shape'):
        result._keras_shape = (x._keras_shape[0], factor*x._keras_shape[1], x._keras_shape[2])

    return result


def up_sampling_2d(x, factors):
    input_shape = x.shape
    output_shape = (input_shape[0], factors[0]*input_shape[1], factors[1]*input_shape[2], input_shape[3])
    output = TT.zeros(output_shape)

    result = TT.set_subtensor(output[:, ::factors[0], ::factors[1], :], x)
    if hasattr(x, '_keras_shape'):
        result._keras_shape = (x._keras_shape[0], factor*x._keras_shape[1], factor*x._keras_shape[2], x._keras_shape[3])
    return result


import tensorflow as tf

def up_sampling_1d_tensorflow(x, factor):
    #print(x)
    #print(type(x))
    # TODO if shape is None:

    # if shape[0] is None:
    # shape = (tf.shape(x)[0], ) + tuple(shape[1:])
    # shape = tf.stack(list(shape))

    shape0 = 1
    input_shape = x.shape
    output_shape = [shape0, factor * input_shape[1].value, input_shape[2].value]
    output = tf.Variable(np.zeros(output_shape), dtype=x.dtype, trainable=False)
    result = output[:, ::factor, :].assign(x)
    """
    # shape0 = tf.placeholder(tf.int32, shape=[])
    shape0 = 1
    # shape0 = tf.shape(x)[0]
    input_shape = x.shape
    output_shape = [shape0, factor * input_shape[1].value, input_shape[2].value]
    # z_val = np.zeros(output_shape)
    z_val = tf.zeros(output_shape)
    output = tf.Variable(z_val, validate_shape=False, dtype=x.dtype, trainable=False)
    result = output[:, ::factor, :].assign(x)
    """
    if hasattr(x, '_keras_shape'):
        result._keras_shape = (x._keras_shape[0], factor*x._keras_shape[1], x._keras_shape[2])

    return result



def up_sampling_2d_tensorflow(x, factors):
    shape0 = 1
    input_shape = x.shape
    output_shape = [shape0, factors[0] * input_shape[1].value, factors[1] * input_shape[2].value, input_shape[3].value]
    output = tf.Variable(np.zeros(output_shape), dtype=x.dtype, trainable=False)
    result = output[:, ::factors[0], ::factors[1], :].assign(x)

    if hasattr(x, '_keras_shape'):
        result._keras_shape = (x._keras_shape[0], factors[0]*x._keras_shape[1], factors[1]*x._keras_shape[2], x._keras_shape[3])

    return result





def interleave_zeros(x, factor, axis):

    shape = K.shape(x)
    #out_shape = [s for s in shape]
    out_shape = [shape[0], shape[1], shape[2], shape[3]]
    out_shape[axis] = factor * out_shape[axis]
    out_order = [n for n in range(1,len(out_shape)+1)]
    out_order.insert(axis+1, 0)
    input_with_zeros = K.stack([x] + (factor - 1)*[K.zeros_like(x)])
    return K.reshape(K.permute_dimensions(input_with_zeros, out_order), out_shape)







class UpSampling1DZeros(Layer):

   def __init__(self, upsampling_factor=2, **kwargs):
      super(UpSampling1DZeros, self).__init__(**kwargs)
      self.upsampling_factor = upsampling_factor
      self.input_spec = InputSpec(ndim=3)

   def compute_output_shape(self, input_shape):
      size = self.upsampling_factor * input_shape[1] if input_shape[1] is not None else None
      return (input_shape[0], size, input_shape[2])

   def call(self, inputs):
      if K.backend() == 'theano':
         return up_sampling_1d_theano(inputs, self.upsampling_factor)
      else:
         shape = K.shape(inputs)
         in_t = K.transpose(inputs)
         return K.reshape(K.transpose(K.stack([in_t, K.zeros_like(in_t)])), [-1,2*shape[1],shape[2]])

   def get_config(self):
      config = {'size': self.size}
      base_config = super(UpSampling1D, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))







class UpSampling1D(Layer):

    def __init__(self, size=2, fill='repeat', **kwargs):
        super(UpSampling1D, self).__init__(**kwargs)
        self.size = int(size)
        self.input_spec = InputSpec(ndim=3)

        fill_modes = {'repeat': self._call_repeat,
                      'zeros': self._call_zeros}

        if fill not in fill_modes.keys():
            raise ValueError(
                'The `mode` argument must be one of "' +
                '", "'.join(fill_modes.keys()) + '". Received: ' + str(fill))

        self.call = fill_modes[fill]

    def compute_output_shape(self, input_shape):
        size = self.size * input_shape[1] if input_shape[1] is not None else None
        return (input_shape[0], size, input_shape[2])

    def _call_repeat(self, inputs):
        output = K.repeat_elements(inputs, self.size, axis=1)
        return output

    def _call_zeros(self, inputs):
        if K.backend() == 'theano':
            return up_sampling_1d_theano(inputs, self.size)
        elif K.backend() == 'tensorflow':
            return up_sampling_1d_tensorflow(inputs, self.size)
        else:
            shape = K.shape(inputs)
            input_with_zeros = K.stack(
                [inputs] + [K.zeros_like(inputs) for _ in range(self.size - 1)])
            return K.reshape(
                K.permute_dimensions(input_with_zeros, (1, 2, 0, 3)),
                [-1, self.size * shape[1], shape[2]])

    def get_config(self):
        config = {'size': self.size}
        base_config = super(UpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling2D(Layer):

    def __init__(self, size=(2, 2), data_format=None, fill='repeat', **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

        fill_modes = {'channels_first_repeat': self._call_repeat,
                      'channels_last_repeat': self._call_repeat,
                      'channels_first_zeros': self._call_channels_first_zeros,
                      'channels_last_zeros': self._call_channels_last_zeros}

        if fill not in ['repeat', 'zeros']:
            raise ValueError(
                'The `mode` argument must be one of "' +
                '", "'.join(fill_modes.keys()) + '". Received: ' + str(fill))

        self.call = fill_modes[data_format + '_' + fill]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def _call_repeat(self, inputs):
        return K.resize_images(inputs, self.size[0], self.size[1],
                               self.data_format)

    def _call_zeros_impl(self, inputs, order1, shape1, order2, shape2):
        input_with_zeros = K.stack(
            [inputs] + [K.zeros_like(inputs) for _ in range(self.size[0] - 1)])
        rows_interleaved = K.reshape(
            K.permute_dimensions(input_with_zeros, order1), shape1)
        rows_interleaved_zeros = K.stack(
            [rows_interleaved] + [K.zeros_like(rows_interleaved) for _ in range(self.size[1] - 1)])
        return K.reshape(
            K.permute_dimensions(rows_interleaved_zeros, order2), shape2)

    def _call_channels_first_zeros(self, inputs):
        shape = K.shape(inputs)
        return self._call_zeros_impl(
            inputs,
            (1, 2, 3, 0, 4), [-1, shape[1], self.size[0] * shape[2], shape[3]],
            (1, 2, 3, 4, 0), [-1, shape[1], self.size[0] * shape[2], self.size[1] * shape[3]])

    def _call_channels_last_zeros(self, inputs):
        out = inputs
        out = interleave_zeros(out, self.size[1], axis=2)
        out = interleave_zeros(out, self.size[0], axis=1)
        return out
        if K.backend() == 'theano':
            return up_sampling_2d(inputs, self.size)
        #elif K.backend() == 'tensorflow':
        #    return up_sampling_2d_tensorflow(inputs, self.size)
        else:
            shape = K.shape(inputs)
            input_with_zeros1 = K.stack([inputs] + (self.size[0]-1) * [K.zeros_like(inputs)])
            input_with_zeros2 = K.stack([input_with_zeros1] + (self.size[1]-1) * [K.zeros_like(input_with_zeros1)])
            return K.reshape(
                K.permute_dimensions(input_with_zeros2, [2,3,1,4,5,0]),
                [-1, self.size[0] * shape[1], self.size[1] * shape[2], shape[3]]
            )
            # return self._call_zeros_impl(
            #     inputs,
            #     (1, 2, 0, 3, 4), [-1, self.size[0] * shape[1], shape[2], shape[3]],
            #     (1, 2, 3, 0, 4), [-1, self.size[0] * shape[1], self.size[1] * shape[2], shape[3]])

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling3D(Layer):

    def __init__(self, size=(2, 2, 2), data_format=None, fill='repeat', **kwargs):
        super(UpSampling3D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.input_spec = InputSpec(ndim=5)

        fill_modes = {'channels_first_repeat': self._call_repeat,
                      'channels_last_repeat': self._call_repeat,
                      'channels_first_zeros': self._call_channels_first_zeros,
                      'channels_last_zeros': self._call_channels_last_zeros}

        if fill not in ['repeat', 'zeros']:
            raise ValueError(
                'The `mode` argument must be one of "' +
                '", "'.join(fill_modes.keys()) + '". Received: ' + str(fill))

        self.call = fill_modes[data_format + '_' + fill]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            dim1 = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            dim2 = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            dim3 = self.size[2] * input_shape[4] if input_shape[4] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    dim1,
                    dim2,
                    dim3)
        elif self.data_format == 'channels_last':
            dim1 = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            dim2 = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            dim3 = self.size[2] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    dim1,
                    dim2,
                    dim3,
                    input_shape[4])

    def _call_repeat(self, inputs):
        return K.resize_volumes(inputs,
                                self.size[0], self.size[1], self.size[2],
                                self.data_format)

    def _call_zeros_impl(self, inputs, order1, shape1, order2, shape2, order3, shape3):
        input_with_zeros = K.stack(
            [inputs] + [K.zeros_like(inputs) for _ in range(self.size[0] - 1)])
        dim1_interleaved = K.reshape(
            K.permute_dimensions(input_with_zeros, order1), shape1)
        dim1_interleaved_zeros = K.stack(
            [dim1_interleaved] + [K.zeros_like(dim1_interleaved) for _ in range(self.size[1] - 1)])
        dim2_interleaved = K.reshape(
            K.permute_dimensions(dim1_interleaved_zeros, order2), shape2)
        dim2_interleaved_zeros = K.stack(
            [dim2_interleaved] + [K.zeros_like(dim2_interleaved) for _ in range(self.size[2] - 1)])
        return K.reshape(
            K.permute_dimensions(dim2_interleaved_zeros, order3), shape3)

    def _call_channels_first_zeros(self, inputs):
        sh = K.shape(inputs)
        up = self.size
        return self._call_zeros_impl(
            inputs,
            (1, 2, 3, 0, 4, 5), [-1, sh[1], up[0] * sh[2], sh[3], sh[4]],
            (1, 2, 3, 4, 0, 5), [-1, sh[1], up[0] * sh[2], up[1] * sh[3], sh[4]],
            (1, 2, 3, 4, 5, 0), [-1, sh[1], up[0] * sh[2], up[1] * sh[3], up[2] * sh[4]])

    def _call_channels_last_zeros(self, inputs):
        sh = K.shape(inputs)
        up = self.size
        return self._call_zeros_impl(
            inputs,
            (1, 2, 0, 3, 4, 5), [-1, up[0] * sh[1], sh[2], sh[3], sh[4]],
            (1, 2, 3, 0, 4, 5), [-1, up[0] * sh[1], up[1] * sh[2], sh[3], sh[4]],
            (1, 2, 3, 4, 0, 5), [-1, up[0] * sh[1], up[1] * sh[2], up[2] * sh[3], sh[4]])

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(UpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))









import time

class Timer:

   def __enter__(self):
      self.t1 = time.time()
      return self

   def __exit__(self,t,v,tb):
      t2 = time.time()
      print("{0:.4f} seconds".format(t2-self.t1))




#input_len = 3
input_len = 2**12
factor = 4

def upsampling_zeros():
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))(input)
   #up2 = UpSampling1DZeros(2)(reshape)
   up2 = UpSampling1D(factor, fill='zeros')(reshape)
   return K.function([input],[up2])

def upsampling_repeat():
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))(input)
   #up2 = L.UpSampling1D(2)(reshape)
   up2 = UpSampling1D(factor, fill='repeat')(reshape)
   return K.function([input],[up2])

def upsampling_zeros_repeat():
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))(input)
   #up2 = UpSampling1DZeros(2)(reshape)
   up2 = UpSampling1D(2, fill='zeros')(reshape)
   conv = L.Conv1D(
      1, padding='same', kernel_size=(factor), use_bias=False,
      weights=[np.array([[[1]*factor]]).T]
   )(up2)
   return K.function([input],[conv])


x = np.arange(input_len).reshape(1,input_len)

trials = 1000


print("--- fill = 'zeros' ---------------------")
f = upsampling_zeros()
#print(f([x])[0][0][:10])

with Timer() as t:
   for n in range(trials):
      f([x])


print("--- fill = 'repeat' --------------------")
f = upsampling_repeat()

with Timer() as t:
   for n in range(trials):
      f([x])


print("--- simulating repeat → zeros + conv ---")
f = upsampling_zeros_repeat()

with Timer() as t:
   for n in range(trials):
      f([x])





size_dim1 = 1024
size_dim2 = 1024
img = np.arange(size_dim1*size_dim2).reshape(1,size_dim1,size_dim2)


def upsampling2d(fill):
   input = L.Input(shape=(size_dim1, size_dim2,))
   reshape = L.Reshape((size_dim1, size_dim2, 1))(input)
   up2 = UpSampling2D(factor, data_format='channels_last', fill=fill)(reshape)
   return K.function([input],[up2])

def upsampling2d_zeros():
   return upsampling2d('zeros')

def upsampling2d_repeat():
   return upsampling2d('repeat')


print("--- 2D --------------------------------")


print("--- fill = 'zeros' --------------------")
f = upsampling2d_zeros()

with Timer() as t:
   for n in range(10):
      f([img])


print("--- fill = 'repeat' --------------------")
f = upsampling2d_repeat()

with Timer() as t:
   for n in range(10):
      f([img])



size_dim1 = 2
size_dim2 = 3
img = np.array( [[1.,2.,3.], [4.,5.,6.]] ).reshape(1,size_dim1, size_dim2)
input = L.Input(shape=(size_dim1, size_dim2,))
reshape = L.Reshape((size_dim1, size_dim2, 1))(input)
up2 = UpSampling2D([3,2], data_format='channels_last', fill='zeros')(reshape)
f = K.function([input],[up2])
print(f([img])[0][:,:,:,0])

# input = L.Input(shape=(4,))
# dense = L.Dense(2, use_bias=False, weights=[np.array([[1,2,3,4],[5,6,7,8]]).T] )
# dense_v = dense(input)
# g = K.function([input],[dense_v])
# print(g([np.array([[1,2,3,4]])]))





"""

size_dim1 = 32
size_dim2 = 32
size_dim3 = 32
vol = np.arange(size_dim1*size_dim2*size_dim3).reshape(1,size_dim1,size_dim2,size_dim3)


def upsampling3d(fill):
   input = L.Input(shape=(size_dim1, size_dim2, size_dim3,))
   reshape = L.Reshape((size_dim1, size_dim2, size_dim3, 1))(input)
   up2 = UpSampling3D(factor, data_format='channels_last', fill=fill)(reshape)
   return K.function([input],[up2])

def upsampling3d_zeros():
   return upsampling3d('zeros')

def upsampling3d_repeat():
   return upsampling3d('repeat')


print("--- 3D --------------------------------")


print("--- fill = 'zeros' --------------------")
f = upsampling3d_zeros()

with Timer() as t:
   for n in range(100):
      f([vol])


print("--- fill = 'repeat' --------------------")
f = upsampling3d_repeat()

with Timer() as t:
   for n in range(100):
      f([vol])





# NUMPY UP-SAMPLTING ---------------------------------------------------------------------------------

# just 2d image
a = np.array([[1,2,3],[4,5,6]])
np.stack([np.stack([a] + 2*[np.zeros([2,3])]), np.zeros([3,2,3])]).transpose(2,1,3,0).reshape(6,6)


# with batches, size = 2
a = np.array([ [[1,2,3],[4,5,6]] , [[9,8,7],[8,7,6]] ]) 
np.stack([np.stack([a] + 2*[np.zeros([2,2,3])]), np.zeros([3,2,2,3])]).transpose(2,3,1,4,0).reshape(2,6,6)


# with features, size = 1
a = np.array([ [[[1],[2],[3]], [[4],[5],[6]]] , [[[9],[8],[7]],[[8],[7],[6]]] ])
np.stack([np.stack([a] + 2*[np.zeros([2,2,3,1])]), np.zeros([3,2,2,3,1])]).transpose(2,3,1,4,5,0).reshape(2,6,6,1)


"""