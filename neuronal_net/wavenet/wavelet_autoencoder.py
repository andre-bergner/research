import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as B

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
      input_shape = inputs.shape
      input_shape[1] *= self.upsampling_factor
      output = np.zeros(input_shape)
      output[:,::self.upsampling_factor,:] = inputs[:,:,:]     # DOES NOT WORK
      return output

   def get_config(self):
      config = {'size': self.size}
      base_config = super(UpSampling1D, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))



def make_analysis_node(kernel):
   return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo(): return make_analysis_node([1,1])
def make_hi(): return make_analysis_node([1,-1])

def make_synth_node(kernel):
   return L.Conv1D(1, kernel_size=(len(kernel)), padding='same', weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo_s(): return make_synth_node([1,1])
def make_hi_s(): return make_synth_node([1,-1])


def build_dyadic_grid(num_levels=3, encoder_size=10, input_len=None):

   # --- create all operators (nodes) ----------------------
   input_len = input_len or 2**num_levels
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))
   synth_slices = []     # the slices for the synthesis network
   right_crop = input_len
   left_crop = 0
   current_level_in = reshape(input)
   out_layers = []
   observables = []
   for _ in range(num_levels):

      # TODO: wavelet weights must be tied!
      hi = make_hi()
      lo = make_lo()
      hi_v = hi(current_level_in)
      lo_v = lo(current_level_in)
      current_level_in = lo_v
      out_layers.append(hi_v)
      observables.append(hi.output)

      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop

   out_layers.append(lo_v)
   observables.append(lo.output)

   flatten = L.Flatten()
   synth_slices.append(L.Cropping1D([left_crop, 0]))

   analysis_layers_v = L.concatenate(out_layers, axis=1)

   encoder = L.Dense(units=input_len, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len, weights=[np.eye(input_len), np.zeros(input_len)])
   reshape2 = L.Reshape((input_len,1))

   decoder_v = reshape2(decoder(encoder(L.Flatten()(analysis_layers_v))))

   synth_slices_v = [l(decoder_v) for l in synth_slices]

   scaling_in = synth_slices_v.pop()
   for _ in range(num_levels):
      detail_in = synth_slices_v.pop()
      observables.append(scaling_in)
      observables.append(detail_in)
      lo = make_lo_s()
      hi = make_hi_s()
      lo_v = lo(L.UpSampling1D(2)(scaling_in))
      hi_v = hi(L.UpSampling1D(2)(detail_in))
      level_synth_v = L.add([lo_v, hi_v])
      scaling_in = level_synth_v

   observables.extend(synth_slices_v)

   model_f = B.function([input], observables)

   return model_f


model_f = build_dyadic_grid(2,input_len=12)
for x in model_f([ np.matrix([1,2,3,4,3,2,1,0,-1,-2,-1,0]) ]):
#model_f = build_dyadic_grid(2,input_len=8)
#for x in model_f([ np.matrix([1,2,3,4,3,2,1,0]) ]):
#model_f = build_dyadic_grid(2,input_len=4)
#for x in model_f([ np.matrix([1,2,0,-1]) ]):
   print(x)

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
