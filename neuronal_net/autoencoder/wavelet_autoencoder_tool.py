#  WORK IN PROGRESS
#  * porting wavelet autoencoder as a tool that can be put around an existing autoencoder

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
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals

"""
h: wavelet
l: scaling
g: synth wavelet
k: synth scaling

    h1------ c ------g1
   /         o         \
  x     h2-- n --g2     y
   \  /      c      \  /
    l1--l2-- a --k2--k1
             t
"""

kernel_size = 2
shared_weights_in_cascade = True
num_features = 1
size = 128
n_latent = size #32


def make_ana_node(down_factor=1):
   #return L.Conv1D(num_features, kernel_size=(kernel_size+1), padding='same', strides=2, use_bias=False, activation=None)
   pad = L.ZeroPadding1D((0,kernel_size-1))
   conv = L.Conv1D(num_features, kernel_size=(kernel_size), padding='valid', strides=2, use_bias=False, activation=None)
   return fun._ >> pad >> conv

def analysis_wavelet_pair():
   return make_ana_node(), make_ana_node()


def make_synth_node(down_factor=1):
   return L.Conv1D(num_features, kernel_size=(kernel_size), padding='same', strides=1, use_bias=False, activation=None)

def synth_wavelet_pair():
   return make_synth_node(), make_synth_node()



def dwt_bank(analysis_wavelet_pair, num_levels=3, input_len=None, num_features=1, shared_weights_in_cascade=False):

   input_len = input_len or 2**num_levels

   def analysis(input):
      reshape = L.Reshape((input_len,1))
      reshaped_input = reshape(input)
      if num_features > 1:
         current_level_in = L.concatenate(num_features*[reshaped_input],axis=2)
      else:
         current_level_in = reshaped_input
      out_layers = []

      casc = tools.CascadeFactory(analysis_wavelet_pair, shared=shared_weights_in_cascade)
      for _ in range(num_levels):
         lo, hi = casc.get()
         lo_v = lo(current_level_in)
         hi_v = hi(current_level_in)
         current_level_in = lo_v
         out_layers.append(hi_v)

      out_layers.append(lo_v)

      return L.concatenate(out_layers, axis=1)


   synth_slices = []     # the slices for the synthesis network
   left_crop = 0
   right_crop = input_len
   for _ in range(num_levels):
      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop
   synth_slices.append(L.Cropping1D([left_crop, 0]))

   def synthesis(input):

      #reshape = L.Reshape((input_len,1))
      #reshaped_input = reshape(input)
      synth_slices_v = [l(input) for l in synth_slices]
      scaling_in = synth_slices_v.pop()
      casc = tools.CascadeFactory(synth_wavelet_pair, shared=shared_weights_in_cascade)

      for _ in range(num_levels):
         detail_in = synth_slices_v.pop()
         lo, hi = casc.get()
         lo_v = lo(Up.UpSampling1DZeros(2)(scaling_in))
         hi_v = hi(Up.UpSampling1DZeros(2)(detail_in))
         level_synth_v = L.add([lo_v, hi_v])
         scaling_in = level_synth_v

      # downmix = L.Conv1D(1, kernel_size=(1), use_bias=False)(level_synth_v)
      # return L.Flatten()(downmix)
      return L.Flatten()(level_synth_v)

   return  fun._ >> analysis, fun._ >> synthesis


def make_model(num_levels=3, n_latent=10, input_len=None):

   input_len = input_len or 2**num_levels
   input = L.Input(shape=(input_len,))

   ana, syn = dwt_bank(analysis_wavelet_pair, num_levels, input_len, num_features, shared_weights_in_cascade)
   return M.Model([input], [(ana >> syn)(input)])

   ### some preliminary attempts to compare wavelet-AE with pure AE:
   ### just AE
   # core_model = F.dense([n_latent], 'tanh') >> F.dense([input_len], 'tanh')
   # return M.Model([input], [core_model(input)])
   ### with wavelet
   # core_model = F.flatten() >> F.dense([n_latent], 'tanh') >> F.dense([input_len,1], 'tanh')
   # return M.Model([input], [(ana >> core_model >> syn)(input)])




model = make_model(5, input_len=size, n_latent=n_latent)

loss_recorder = tools.LossRecorder()
np.random.seed(1337)
data = test_signals.decaying_sinusoids(size, num_signals=512, num_modes=3)

model.compile(optimizer=keras.optimizers.Adam(), loss='mae')
model.summary()
model.fit(data, data, batch_size=64, epochs=300, verbose=0, callbacks=[tools.Logger(),loss_recorder])


from pylab import *

def plot_input_vs_approx(n, data=data):
   figure()
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

#for w in model.get_weights():
#   print(w)

semilogy(loss_recorder.losses)
plot_input_vs_approx(0)


import scipy.signal as ss

def plot_transfer_function(weight_id,channel1=0,channel2=0):
   w,H = ss.freqz(model.get_weights()[weight_id][:,channel1,channel2], 1024)
   plot(w, abs(H))

#from keras.utils import plot_model
#plot_model(model, to_file='wavelet_autoencoder.png')
