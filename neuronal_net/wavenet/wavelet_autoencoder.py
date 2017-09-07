import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as B

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


def make_analysis_node(kernel):
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), use_bias=True)
   return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), use_bias=False)
   #return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo(): return make_analysis_node([1,1])
def make_hi(): return make_analysis_node([1,-1])

def make_synth_node(kernel):
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', use_bias=True)
   return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', use_bias=False)
   #return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo_s(): return make_synth_node([1,1])
def make_hi_s(): return make_synth_node([1,-1])


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
   hi = make_hi()    # these weights are shared
   lo = make_lo()    # move into loop to have weights per layer
   for _ in range(num_levels):

      #hi = make_hi()
      #lo = make_lo()
      hi_v = hi(current_level_in)
      lo_v = lo(current_level_in)
      current_level_in = lo_v
      out_layers.append(hi_v)
      #observables.append(hi.output)

      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop

   out_layers.append(lo_v)
   #observables.append(lo.output)

   flatten = L.Flatten()
   synth_slices.append(L.Cropping1D([left_crop, 0]))

   analysis_layers_v = L.concatenate(out_layers, axis=1)

   encoder = L.Dense(units=encoder_size)#, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len)#, weights=[np.eye(input_len), np.zeros(input_len)])
   reshape2 = L.Reshape((input_len,1))

   decoder_v = reshape2(decoder(encoder(L.Flatten()(analysis_layers_v))))
   #decoder_v = analysis_layers_v      # no en/de-coder

   synth_slices_v = [l(decoder_v) for l in synth_slices]

   scaling_in = synth_slices_v.pop()
   hi = make_hi_s()
   lo = make_lo_s()
   for _ in range(num_levels):
      detail_in = synth_slices_v.pop()
      #observables.append(scaling_in)
      #observables.append(detail_in)
      #hi = make_hi_s()
      #lo = make_lo_s()
      hi_v = hi(L.UpSampling1D(2)(detail_in))
      lo_v = lo(L.UpSampling1D(2)(scaling_in))
      level_synth_v = L.add([lo_v, hi_v])
      scaling_in = level_synth_v

   output = L.Flatten()(level_synth_v)

   #return B.function([input], [output])

   #observables.append(level_synth_v)
   #model_f = B.function([input], observables)
   #return model_f

   return M.Model(inputs=input, outputs=output)



size = 32

model = build_dyadic_grid(5, input_len=size, encoder_size=32)
model.compile(optimizer='sgd', loss='mean_absolute_error')


def make_test_signals(size, num_signals=200, num_modes=5):

   def make_mode(damp, freq, phase):
      time = np.arange(size)
      return np.cos(phase + freq*time) * np.exp(damp*time)

   signals = [
      np.sum([
         make_mode( -d, np.pi*f, np.pi*p )
         for d,f,p in np.random.rand(num_modes,3)],
         axis=0)
      for _ in range(num_signals)
   ]

   return np.array(signals)

loss_recorder = tools.LossRecorder()
data = make_test_signals(size)
model.fit(data, data, batch_size=20, epochs=300, verbose=0,
   callbacks=[tools.Logger(),loss_recorder])


from keras.utils import plot_model
plot_model(model, to_file='wavelet_autoencoder.png')


from pylab import *

prediction = model.predict(data[0:1])
plot(data[0])
plot(prediction.T)



"""
hi1_o, hi2_o, lo2_o, net_o, hi1_s, hi2_s, lo2_s = model_f([ np.matrix([1,2,3,4,3,2,1,0]) ])

print( "hi1: {}".format(hi1_o.flatten()) )
print( "hi2: {}".format(hi2_o.flatten()) )
print( "lo2: {}".format(lo2_o.flatten()) )
#print( "all: {}".format(net_o.T.flatten()) )
print( "pyramids:\n{}".format(net_o[:,:,0]) )
print( "hi1 syn: {}".format(hi1_s.flatten()) )
print( "hi2 syn: {}".format(hi2_s.flatten()) )
print( "lo2 syn: {}".format(lo2_s.flatten()) )
"""
