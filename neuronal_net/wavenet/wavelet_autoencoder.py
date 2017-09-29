import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
import keras.regularizers as regularizers

import tools

"""
h: wavelet
l: scaling
g: synth wavelet
k: synth scaling
      x
    /  \
   h1  l1
   |  / \
   | h2 l2
   \ |  /
   concat
   / |  \
  | g2 k2
  |  \ /
  g1 k1
  \  /
   y
"""



from theano import tensor as TT

def up_sampling_1d(x, factor):
    input_shape = x.shape
    output_shape = (input_shape[0], factor*input_shape[1], input_shape[2])
    output = TT.zeros(output_shape)

    result = TT.set_subtensor(output[:, ::factor, :], x)
    if hasattr(x, '_keras_shape'):
        result._keras_shape = (x._keras_shape[0], factor*x._keras_shape[1], x._keras_shape[2])
    return result


def interleave_zeros(x, factor, axis):

    shape = K.shape(x)
    out_shape = [shape[0], shape[1], shape[2], shape[3]]
    out_shape[axis] = factor * out_shape[axis]
    out_order = [n for n in range(1,len(out_shape)+1)]
    out_order.insert(axis+1, 0)
    input_with_zeros = K.stack([x] + (factor - 1)*[K.zeros_like(x)])
    return K.reshape(K.permute_dimensions(input_with_zeros, out_order), out_shape)



from keras.engine.topology import Layer
from keras.engine import InputSpec

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
         return up_sampling_1d(inputs, self.upsampling_factor)
      else:
         return interleave_zeros(inputs, self.upsampling_factor, axis=1)

   def get_config(self):
      config = {'size': self.size}
      base_config = super(UpSampling1D, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))



kernel_size = 7
shared_weights_in_cascade = True
use_same_kernel_for_analysis_and_synthesis = False
num_features = 4
encoder_size = 10

def make_analysis_node(down_factor=1):
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), use_bias=False, activation='tanh')
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])
   #return L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=2, use_bias=False, activation='tanh')
   conv1 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=2, use_bias=False, activation='tanh')
   conv2 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=down_factor, use_bias=False, activation='tanh')
   #conv3 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=down_factor, use_bias=False, activation='tanh')

   def chain(input):
      return conv2(conv1(input))
      #return conv3(conv2(conv1(input)))

   return chain


def make_lo(): return make_analysis_node(1)
def make_hi(): return make_analysis_node(2)

def make_synth_node(down_factor=1):
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', use_bias=False, activation='tanh')
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', weights=[np.array([[kernel]]).T, np.zeros(1)])
   #return L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation='tanh')

   conv1 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation='tanh')
   conv2 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation='tanh')

   def chain(input):
      return conv2(conv1(input))

   return chain

def make_lo_s(): return make_synth_node()
def make_hi_s(): return make_synth_node()




class CascadeFactory:

   def __init__(self, scaling_factory, wavelet_factory, shared=True):
      if shared:
         scaling_function = scaling_factory();
         wavelet_function = wavelet_factory();
         self.scaling_factory = lambda: scaling_function
         self.wavelet_factory = lambda: wavelet_function
      else:
         self.scaling_factory = scaling_factory
         self.wavelet_factory = wavelet_factory

   def scaling(self):
      return self.scaling_factory()

   def wavelet(self):
      return self.wavelet_factory()




def build_codercore(input_len, encoder_size):

   activation = None
   #activation = 'tanh'
   #encoder = L.Dense(units=encoder_size, activation=activation, kernel_regularizer=regularizers.l1(10e-4))#, weights=[np.eye(input_len), np.zeros(input_len)])
   #decoder = L.Dense(units=input_len, activation=activation, kernel_regularizer=regularizers.l1(10e-4))#, weights=[np.eye(input_len), np.zeros(input_len)])
   encoder = L.Dense(units=encoder_size, activation=activation, activity_regularizer=regularizers.l1(0.001))#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len*num_features, activation=activation)
   #encoder = L.Dense(units=encoder_size, activation=activation)
   #decoder = L.Dense(units=input_len, activation=activation)
   reshape2 = L.Reshape((input_len, num_features))
   reshape2.activity_regularizer = keras.regularizers.l1(l=0.001)

   def chain(input):
      return reshape2(decoder(encoder(L.Flatten()(input))))

   return chain



def build_codercore2(input_len, encoder_size):

   activation = None
   encoder = L.Dense(units=encoder_size, activation=activation)#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len*num_features, activation=activation)
   reshape2 = L.Reshape((input_len, num_features))

   def chain(input):
      return reshape2(decoder(encoder(L.Flatten()(input))))

   return chain


def build_dyadic_grid(num_levels=3, encoder_size=10, input_len=None):

   input_len = input_len or 2**num_levels
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))
   synth_slices = []     # the slices for the synthesis network
   right_crop = input_len
   left_crop = 0
   reshaped_input = reshape(input)
   if num_features > 1:
      current_level_in = L.concatenate(num_features*[reshaped_input],axis=2)
   else:
      current_level_in = reshaped_input
   out_layers = []
   #observables = []
   casc = CascadeFactory(make_lo, make_hi, shared=shared_weights_in_cascade)
   for _ in range(num_levels):

      lo_v = casc.scaling()(current_level_in)
      hi_v = casc.wavelet()(current_level_in)
      current_level_in = lo_v
      out_layers.append(hi_v)
      #observables.append(hi.output)

      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop

   out_layers.append(lo_v)
   synth_slices.append(L.Cropping1D([left_crop, 0]))
   #observables.append(lo.output)

   analysis_layers_v = L.concatenate(out_layers, axis=1)

   coder = build_codercore(input_len, encoder_size)
   decoder_v = coder(analysis_layers_v)
   #decoder_v = analysis_layers_v      # no en/de-coder

   synth_slices_v = [l(decoder_v) for l in synth_slices]

   scaling_in = synth_slices_v.pop()

   if use_same_kernel_for_analysis_and_synthesis:
      casc.scaling().strides = [1]  # HACK: make the model weights shareable
      casc.wavelet().strides = [1]  # but having different strides per node
   else:
      casc = CascadeFactory(make_lo_s, make_hi_s, shared=shared_weights_in_cascade)

   for _ in range(num_levels):
      detail_in = synth_slices_v.pop()
      #observables.append(scaling_in)
      #observables.append(detail_in)
      lo_v = casc.scaling()(UpSampling1DZeros(2)(scaling_in))
      hi_v = casc.wavelet()(UpSampling1DZeros(2)(detail_in))
      level_synth_v = L.add([lo_v, hi_v])
      scaling_in = level_synth_v

   downmix = L.Conv1D(1, kernel_size=(1), use_bias=False)(level_synth_v)
   output = L.Flatten()(downmix)
   #output = L.Flatten()(level_synth_v)

   #return K.function([input], [output])

   #observables.append(level_synth_v)
   #model_f = K.function([input], observables)
   #return model_f

   model = M.Model(inputs=input, outputs=output)
   #model.layers[4].activity_regularizer = keras.regularizers.l1(l=0.001)
   return model



size = 32

model = build_dyadic_grid(5, input_len=size, encoder_size=encoder_size)
model.compile(optimizer=keras.optimizers.SGD(lr=1), loss='mean_absolute_error')


def make_test_signals(size, num_signals=200, num_modes=5):

   def make_mode(damp, freq, phase):
      time = np.arange(size)
      return np.cos(phase + freq*time) * np.exp(damp*time)

   signals = [
      np.sum([
         make_mode( -0.1*d, np.pi*f, np.pi*p )
         for d,f,p in np.random.rand(num_modes,3)],
         axis=0)
      for _ in range(num_signals)
   ]

   return np.array(signals) / np.max(np.abs(signals))

loss_recorder = tools.LossRecorder()
data = make_test_signals(size)
model.fit(data, data, batch_size=20, epochs=500, verbose=0,
   callbacks=[tools.Logger(),loss_recorder])


from keras.utils import plot_model
plot_model(model, to_file='wavelet_autoencoder.png')


from pylab import *

def plot_input_vs_approx(n, data=data):
   plot(data[n], 'k')
   plot(model.predict(data[n:n+1]).T, 'r')

def plot_all_diffs(ns=range(len(data))):
   figure()
   for n in ns:
      y = model.predict(data[n:n+1])[0]
      plot(data[n]-y, 'k', alpha=0.3)

def mean_diff():
   return np.mean(
      [data[n] - model.predict(data[n:n+1])[0] for n in range(len(data))],
      axis=0
   )

for w in model.get_weights():
   print(w)

plot_input_vs_approx(0)


import scipy.signal as ss

def plot_transfer_function(weight_id,channel1=0,channel2=0):
   w,H = ss.freqz(model.get_weights()[weight_id][:,channel1,channel2], 1024)
   plot(w, abs(H))

# plotting code en/decoder matrix
# imshow(abs(dot(model.get_weights()[2],model.get_weights()[4])))


