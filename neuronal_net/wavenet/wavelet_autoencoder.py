import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
import keras.regularizers as regularizers

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import upsampling as Up
from keras_tools import functional as fun

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

kernel_size = 2
shared_weights_in_cascade = True
use_same_kernel_for_analysis_and_synthesis = False
num_features = 4
encoder_size = 10

def make_analysis_node(down_factor=1):
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), use_bias=False, activation='tanh')
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])
   #return L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=2, use_bias=False, activation='tanh')
   conv1 = L.Conv1D(num_features, kernel_size=(kernel_size+1), padding='same', strides=2, use_bias=False, activation='tanh')
   conv2 = L.Conv1D(num_features, kernel_size=(kernel_size+1), padding='same', strides=down_factor, use_bias=False, activation='tanh')
   #conv3 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=down_factor, use_bias=False, activation='tanh')

   return fun.Input() >> conv1 >> conv2 #>> conv3


def analysis_scaling_node(): return make_analysis_node(1)
def analysis_wavelet_node(): return make_analysis_node(2)

def make_synth_node(down_factor=1):
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', use_bias=False, activation='tanh')
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', weights=[np.array([[kernel]]).T, np.zeros(1)])
   #return L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation='tanh')

   conv1 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation='tanh')
   conv2 = L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation='tanh')

   return fun.Input() >> conv1 >> conv2

def synth_scaling_node(): return make_synth_node()
def synth_wavelet_node(): return make_synth_node()




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
   encoder = L.Dense(units=encoder_size, activation=activation, activity_regularizer=regularizers.l1(0.0001))#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len*num_features, activation=activation)
   #encoder = L.Dense(units=encoder_size, activation=activation)
   #decoder = L.Dense(units=input_len, activation=activation)
   reshape2 = L.Reshape((input_len, num_features))
   reshape2.activity_regularizer = keras.regularizers.l1(l=0.0001)

   return fun.Input() >> L.Flatten() >> encoder >> decoder >> reshape2


def build_codercore2(input_len, encoder_size):

   activation = None
   encoder = L.Dense(units=encoder_size, activation=activation)#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len*num_features, activation=activation)
   reshape2 = L.Reshape((input_len, num_features))

   return fun.Input() >> L.Flatten() >> encoder >> decoder >> reshape2


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
   casc = CascadeFactory(analysis_scaling_node, analysis_wavelet_node, shared=shared_weights_in_cascade)
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
      casc = CascadeFactory(synth_scaling_node, synth_wavelet_node, shared=shared_weights_in_cascade)

   for _ in range(num_levels):
      detail_in = synth_slices_v.pop()
      #observables.append(scaling_in)
      #observables.append(detail_in)
      lo_v = casc.scaling()(Up.UpSampling1DZeros(2)(scaling_in))
      hi_v = casc.wavelet()(Up.UpSampling1DZeros(2)(detail_in))
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
model.compile(optimizer=keras.optimizers.SGD(lr=.02), loss='mean_absolute_error')


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


#from keras.utils import plot_model
#plot_model(model, to_file='wavelet_autoencoder.png')


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


