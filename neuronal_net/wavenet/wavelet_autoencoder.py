import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as B
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
      return up_sampling_1d(inputs, self.upsampling_factor)
      #shape = B.shape(inputs)
      #in_t = B.transpose(inputs)
      #return B.reshape(B.transpose(B.stack([in_t, B.zeros_like(in_t)])), [-1,2*shape[1],shape[2]])

   def get_config(self):
      config = {'size': self.size}
      base_config = super(UpSampling1D, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))







def make_analysis_node(kernel):
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), use_bias=False, activation='tanh')
   return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), use_bias=False)
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo(): return make_analysis_node([1,1])
def make_hi(): return make_analysis_node([1,-1])

def make_synth_node(kernel):
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', use_bias=False, activation='tanh')
   return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', use_bias=False)
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo_s(): return make_synth_node([1,1])
def make_hi_s(): return make_synth_node([1,-1])


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
   encoder = L.Dense(units=encoder_size, activation=activation, activity_regularizer=regularizers.l1(10e-4))#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len, activation=activation)
   #encoder = L.Dense(units=encoder_size, activation=activation)
   #decoder = L.Dense(units=input_len, activation=activation)
   reshape2 = L.Reshape((input_len,1))
   reshape2.activity_regularizer = keras.regularizers.l1(l=0.0001)

   def chain(input):
      return reshape2(decoder(encoder(L.Flatten()(input))))

   return chain




def build_codercore2(input_len, encoder_size):

   activation = None
   encoder = L.Dense(units=encoder_size, activation=activation)#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len, activation=activation)
   reshape2 = L.Reshape((input_len,1))
   #reshape2.activity_regularizer = keras.regularizers.l1(l=0.0001)

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
   current_level_in = reshape(input)
   out_layers = []
   #observables = []
   casc = CascadeFactory(make_lo, make_hi, shared=True)
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

   coder = build_codercore2(input_len, encoder_size)
   decoder_v = coder(analysis_layers_v)
   #decoder_v = analysis_layers_v      # no en/de-coder

   synth_slices_v = [l(decoder_v) for l in synth_slices]

   scaling_in = synth_slices_v.pop()
   casc = CascadeFactory(make_lo_s, make_hi_s, shared=True)
   for _ in range(num_levels):
      detail_in = synth_slices_v.pop()
      #observables.append(scaling_in)
      #observables.append(detail_in)
      lo_v = casc.scaling()(UpSampling1DZeros(2)(scaling_in))
      hi_v = casc.wavelet()(UpSampling1DZeros(2)(detail_in))
      level_synth_v = L.add([lo_v, hi_v])
      scaling_in = level_synth_v

   output = L.Flatten()(level_synth_v)

   #return B.function([input], [output])

   #observables.append(level_synth_v)
   #model_f = B.function([input], observables)
   #return model_f

   model = M.Model(inputs=input, outputs=output)
   #model.layers[4].activity_regularizer = keras.regularizers.l1(l=0.001)
   return model



size = 32

model = build_dyadic_grid(5, input_len=size, encoder_size=26)
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

for w in model.get_weights():
   print(w)

plot_input_vs_approx(0)

# plotting code en/decoder matrix
# imshow(abs(dot(model.get_weights()[2],model.get_weights()[4])))


