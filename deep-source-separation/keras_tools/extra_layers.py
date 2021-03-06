import keras
import keras.backend as K
import keras.layers as L
from keras.layers import Layer, InputSpec


class SliceLike:

    def __getitem__(self, idx):
        return idx

SLICE_LIKE = SliceLike()


class SliceFactory:

    def __getitem__(self, idx):
        return Slice(idx)

SLICE = SliceFactory()


class MetaSlice(type):

    def __getitem__(self, idx):
        return Slice(idx)


class Slice(Layer, metaclass=MetaSlice):
    """Slice layer

    It slice the input tensor with the provided slice tuple

    # Arguments
        slices: int tuple of ints or slices, as many as the input as dimensions

    # Input shape
        ND tensor with shape `(batch, dim1, ..., dimN, features)`

    # Output shape
        ND tensor with shape `(batch, dim1, ..., dimN, features) ^ slices`
    """

    def __init__(self, slices=None, activity_regularizer=None, **kwargs):
        super(Slice, self).__init__(**kwargs)
        # self.slice = conv_utils.normalize_tuple(slice, 2, 'cropping')

        # plaidML requires int's for slicing, from numpy we get int64 types, though.
        as_int = lambda i: None if i == None else int(i)
        as_int_slice = lambda s: slice(as_int(s.start), as_int(s.stop), as_int(s.step))
        slices = tuple([as_int_slice(s) for s in slices])

        assert type(slices) == tuple
        assert all(type(s) == slice for s in slices)
        # TODO elements might be int --> reduces ndim  -->  forbid or allow ???
        self.slices = slices
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(ndim=len(slices))

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == len(self.slices)

        def compute_output_slice(input_length, slc):
            if input_length is not None:
                # TODO handle negative indices
                if  slc.stop is not None:
                    length = slc.stop
                else:
                    length = input_length
                if  slc.start is not None:
                    length -= slc.start
                if  slc.step is not None:
                    length /= slc.step        # TODO round up / down?
                return length
            else:
                return None

        return tuple(
            compute_output_slice(input_length, slc)
            for input_length, slc in zip(input_shape, self.slices)
        )

    def call(self, inputs):
        return inputs[self.slices]

    def get_config(self):
        config = {  'slices': self.slices
                 ,  'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
                 }
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class AppendDimension(L.Layer):

   """AppendDimension layer

   AppendDimension extends the dimensionality of the input tensor by one.

   # Input shape
     ND tensor with shape `(batch, dim1, ..., dimN)`

   # Output shape
     ND tensor with shape `(batch, dim1, ..., dimN, 1)
   """

   def __init__(self, axis=-1, *args, **kwargs):
      super(AppendDimension, self).__init__(*args, **kwargs)
      self.axis = axis

   def compute_output_shape(self, input_shape):
      n = self.axis
      if n < 0:
         n = self.axis % len(input_shape) + 1
      input_shape_list = list(input_shape)
      input_shape_list.insert(n, 1)
      return tuple(input_shape_list)

   def call(self, inputs, training=None):
      return K.expand_dims(inputs, self.axis)


class RemoveDimension(L.Layer):

   """RemoveDimension layer

   RemoveDimension removes a specified dimension of the input tensor.
   The specified dimension/axis must be of size 1!

   # Input shape
     ND tensor with shape `(batch, dim1, 1, ..., dimN)`

   # Output shape
     ND tensor with shape `(batch, dim1, ..., dimN)
   """

   def __init__(self, axis=-1, *args, **kwargs):
      super(RemoveDimension, self).__init__(*args, **kwargs)
      self.axis = axis

   def compute_output_shape(self, input_shape):
      n = self.axis
      input_shape_list = list(input_shape)
      if input_shape_list[n] != 1:
         raise "Dimension to remove must have size 1."
      del input_shape_list[n]
      return tuple(input_shape_list)

   def call(self, inputs, training=None):
      return K.squeeze(inputs, self.axis)


def stack(tensors):
   return L.concatenate([AppendDimension()(t) for t in tensors])

"""
class Stack(L.Layer):

    def __init__(self, *args, **kwargs):
        super(Stack, self).__init__(*args, **kwargs)

    def compute_output_shape(self, input_shape):
        size = input_shape[1] // self.downsampling_factor if input_shape[1] is not None else None
        return (input_shape[0], size, input_shape[2])

    def call(self, inputs):
        return K.stack(inputs)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(DownSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""

class DecayingGaussianNoise(L.GaussianNoise):
   """Apply additive zero-centered Gaussian noise.

   This is useful to mitigate overfitting
   (you could see it as a form of random data augmentation).
   Gaussian Noise (GS) is a natural choice as corruption process
   for real valued inputs.

   As it is a regularization layer, it is only active at training time.

   This is an extension to GaussianNoise that allows the noise to decay
   over training time. For some applications a stronger noise in the early
   training phase is better but can be reduced over time.

   # Arguments
     stddev: float, standard deviation of the noise distribution.
     decay: decay factor of stddev - after each training step stddev is multiplied by this
     final_stddev: final value for stddev
     correlation_dims: specify a list of numbers of dimensions that should be correlated,
         e.g. correlation_dims=[2, 3] means that the first two dimensions and the following
         three dimensions get the same value added.

   # Input shape
     Arbitrary. Use the keyword argument `input_shape`
     (tuple of integers, does not include the samples axis)
     when using this layer as the first layer in a model.

   # Output shape
     Same shape as input.
   """

   def __init__(self, stddev, decay=0, final_stddev=0, correlation_dims=None, **kwargs):
      super(DecayingGaussianNoise, self).__init__(stddev, **kwargs)
      self.decay = decay
      self.final_stddev = final_stddev
      self.correlation_dims = correlation_dims

   def build(self, input_shape):
      self.factor = K.variable(self.stddev, name="factor")
      super(DecayingGaussianNoise, self).build(input_shape)

   def call(self, inputs, training=None):
      self.add_update([(
         self.factor,   # value to be updated
         self.factor + self.decay * (self.final_stddev - self.factor)   # update expression
      )])

      if self.correlation_dims:
         if self.correlation_dims == 'all':
            def noised():
               noise = K.ones_like(inputs) * K.random_normal(shape=1, mean=0., stddev=1.)
               return inputs + self.factor * noise
         else:
            if hasattr(inputs, '_keras_shape'):
               assert( sum(self.correlation_dims) == inputs._keras_shape[-1] )
            def noised():
               noise = K.concatenate([
                  K.repeat_elements(
                     K.random_normal(shape=1, mean=0., stddev=1.), rep, axis=-1)
                  for rep in self.correlation_dims
               ])
               return inputs + self.factor * noise
      else:
         def noised():
            return inputs + self.factor * K.random_normal(
               shape=K.shape(inputs),  mean=0., stddev=1.)
      return K.in_train_phase(noised, inputs, training=training)

   def get_config(self):
      config = {'decay': self.decay}
      base_config = super(DecayingGaussianNoise, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))




class VariationalEncoder(L.Layer):

   def __init__(self, latent_size, data_size, beta=1, no_sampling=False, *args, **kwargs):
      super(VariationalEncoder, self).__init__(*args, **kwargs)
      self.latent_size = latent_size
      self.data_size = data_size
      self.beta = beta
      self.no_sampling = no_sampling

   def compute_output_shape(self, input_shape):
      return (input_shape[0], self.latent_size)

   def call(self, inputs, training=None):

      def reparameterization(args):
         mu, log_sigma = args
         epsilon = K.random_normal(shape=K.shape(mu))
         sigma = K.exp(0.5 * log_sigma)
         if self.no_sampling:
            return mu + sigma
         else:
            return K.in_train_phase(mu + sigma * epsilon, mu + sigma, training=training)

      h = inputs
      mu = L.Dense(self.latent_size, name='mu')(h)
      log_sigma = L.Dense(self.latent_size, name='log_sigma')(h)
      z = L.Lambda(reparameterization, output_shape=(self.latent_size,), name='z')([mu, log_sigma])

      kl_div = -.5 * K.mean(1 + log_sigma - K.square(mu) - K.exp(log_sigma))
      self.add_loss(kl_div * self.beta * self.latent_size / self.data_size)

      self.mu = mu
      self.log_sigma = log_sigma

      return z




if K.backend() == 'theano':

    from theano import tensor as TT

    def fft(x):
        y = TT.fft.rfft(x)
        if hasattr(x, '_keras_shape'):
            y._keras_shape = (x._keras_shape[0], x._keras_shape[1]/2 + 1, 2)
        return y

    def power_spectrum(x):
        y = TT.fft.rfft(x)
        y = y[:,:,0]**2 + y[:,:,1]**2
        # if hasattr(x, '_keras_shape'):
        #     y._keras_shape = (x._keras_shape[0], x._keras_shape[1]/2 + 1)
        return y

else:

    def fft(x):
        y = tf.fft(x)
        # ...
        return y


class FourierTrafo(Layer):
    """Fourier Transform layer

    Computes a (fast) fourier transform of the input tensor

    # Arguments
        tensor

    # Input shape
        ND tensor with shape `(batch, dim1, ..., dimN, features)`

    # Output shape
        ND tensor with shape `(batch, dim1, ..., dimN, features) ^ slices`
    """

    def __init__(self, **kwargs):
        super(FourierTrafo, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]/2 + 1, 2)

    def call(self, inputs):
        return fft(inputs)






import os
os.environ['KERAS_BACKEND'] = 'theano'

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
import numpy as np


if K.backend() == 'theano':

    import theano
    from theano import tensor as T

    # http://deeplearning.net/software/theano/library/gradient.html#theano.gradient.jacobian
    def jacobian(y, x):
       return theano.gradient.jacobian(y[0,:],x)
       # TODO assert that shape of y is (1,N) or (N,1)
       # TODO pick right dimension
       # J, updates = theano.scan(
       #     lambda i, y, x : T.grad(y[0,i], x),
       #     sequences = T.arange(y.shape[1]),
       #     non_sequences = [y, x]
       # )
       # return J

elif K.backend() == 'tensorflow':

    import tensorflow as tf

    # https://github.com/tensorflow/tensorflow/issues/675

    def jacobian(y, x, n=1):
        y_list = tf.unstack(y, num = n)
        jacobian_list = [
            [tf.gradients(y_, x)[0][i] for y_ in tf.unstack(y_list[i])]
            for i in range(n)
        ] # list [grad(y0, x), grad(y1, x), ...]
        return tf.stack(jacobian_list)

else:

    print(K.backend() + ' backend currently not supported.')




def soft_relu(x, alpha=0.0):
   return (0.5+alpha) * x  +  (0.5-alpha) * K.sqrt(1.0 + x*x)

def tanhx(x, alpha=0.0):
   return (1.-alpha) * K.tanh(x) + alpha * x

keras.utils.generic_utils.get_custom_objects().update({
    'soft_relu': L.Activation(soft_relu),
    'tanhx': L.Activation(tanhx)
})





class LayerNorm1D(Layer):

    # https://arxiv.org/pdf/1607.06450.pdf
    # https://github.com/keras-team/keras/issues/3878

    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=keras.initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True,)

        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
