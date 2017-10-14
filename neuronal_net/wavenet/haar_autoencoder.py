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
from keras_tools import extra_layers as XL

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
encoder_size = 32

def make_analysis_node(num_features=1):
   # This one works as expected but only for kernel_size == 2:
   # return L.Conv1D(1, kernel_size=(kernel_size), strides=2, use_bias=False, activation=None)

   # This one is a workaround that support kernel_size == 2 as well but introduces an
   # unnecessary addiotnal parameter in the kernel, that should be zero
   # return L.Conv1D(1, kernel_size=(kernel_size+1), padding='same', strides=2, use_bias=False, activation=None)

   # This fixes a symmetry issue for the wavelet auto-encoder to work.
   # It reuires a 'same'-convolution but with zeros added at the end instead of the beginning.
   pad = L.ZeroPadding1D((0,kernel_size-1))
   conv = L.Conv1D(num_features, kernel_size=(kernel_size), padding='valid', strides=2, use_bias=False, activation=None)

   return fun.Input() >> pad >> conv

def analysis_scaling_node(): return make_analysis_node()
def analysis_wavelet_node(): return make_analysis_node()

def analysis_wavelet_pair():
   node = make_analysis_node(2)
   take_feat_0 = XL.Slice(XL.SLICE_LIKE[:,:,0:1])
   take_feat_1 = XL.Slice(XL.SLICE_LIKE[:,:,1:2])

   def splitter(x):
      node_v = node(x)
      return (take_feat_0(node_v), take_feat_1(node_v))

   return splitter


def make_synth_node():
   return L.Conv1D(1, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation=None)

def synth_scaling_node(): return make_synth_node()
def synth_wavelet_node(): return make_synth_node()

def synth_analysis_wavelet_pair():
   return make_synth_node(), make_synth_node()



class CascadeFactory:

   def __init__(self, factory, shared=True):
      if shared:
         artefact = factory();
         self.factory = lambda: artefact
      else:
         self.factory = factory

   def get(self):
      return self.factory()




def build_codercore(input_len, encoder_size):

   activation = None
   encoder = L.Dense(units=encoder_size, activation=activation, activity_regularizer=regularizers.l1(0.00001))#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len, activation=activation)
   reshape2 = L.Reshape((input_len, 1))

   return fun.Input() >> L.Flatten() >> encoder >> decoder >> reshape2



def build_dyadic_grid(num_levels=3, encoder_size=32, input_len=None):

   input_len = input_len or 2**num_levels
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))
   synth_slices = []     # the slices for the synthesis network
   right_crop = input_len
   left_crop = 0
   current_level_in = reshape(input)
   out_layers = []
   casc = CascadeFactory(analysis_wavelet_pair, shared=shared_weights_in_cascade)
   for _ in range(num_levels):

      lo_v, hi_v = casc.get()(current_level_in)
      current_level_in = lo_v
      out_layers.append(hi_v)

      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop

   out_layers.append(lo_v)
   synth_slices.append(L.Cropping1D([left_crop, 0]))

   analysis_layers_v = L.concatenate(out_layers, axis=1)

   coder = build_codercore(input_len, encoder_size)
   #decoder_v = coder(analysis_layers_v)
   decoder_v = analysis_layers_v      # no en/de-coder

   synth_slices_v = [l(decoder_v) for l in synth_slices]

   scaling_in = synth_slices_v.pop()

   if use_same_kernel_for_analysis_and_synthesis:
      casc.scaling().strides = [1]  # HACK: make the model weights shareable
      casc.wavelet().strides = [1]  # but having different strides per node
   else:
      casc = CascadeFactory(synth_analysis_wavelet_pair, shared=shared_weights_in_cascade)

   for _ in range(num_levels):
      detail_in = synth_slices_v.pop()
      lo, hi = casc.get()
      lo_v = lo(Up.UpSampling1DZeros(2)(scaling_in))
      hi_v = hi(Up.UpSampling1DZeros(2)(detail_in))
      level_synth_v = L.add([lo_v, hi_v])
      scaling_in = level_synth_v

   output = L.Flatten()(level_synth_v)

   model = M.Model(inputs=input, outputs=output)
   #model.layers[4].activity_regularizer = keras.regularizers.l1(l=0.001)
   return model


size = 32

model = build_dyadic_grid(5, input_len=size, encoder_size=encoder_size)
model.compile(optimizer=keras.optimizers.SGD(lr=.01), loss='mean_absolute_error')
# model.summary()  


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
#plot_model(model, to_file='haar_autoencoder.png')


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
