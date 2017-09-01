import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as B

"""
      x
    /  \
   h1  l1
   |  / \
   | h2 l2
   \ |  /
   concat
"""

def make_haar_kernel(kernel):
   return L.Conv1D(1, kernel_size=(2), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo(): return make_haar_kernel([1,1])
def make_hi(): return make_haar_kernel([1,-1])


def build_dyadic_grid(num_levels=3, encoder_size=10, input_len=None):

   # --- create all operators (nodes) ----------------------
   input_len = input_len or 2**num_levels
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))
   his = []
   los = []
   synth_slices = []     # the slices for the synthesis network
   right_crop = 2**num_levels
   left_crop = 0
   for _ in range(num_levels):
      his.append(make_hi())
      los.append(make_lo())
      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop
   flatten = L.Flatten()
   right_crop >>= 1
   synth_slices.append(L.Cropping1D([left_crop, right_crop]))

   # --- connect nodes to graph ----------------------------
   out_layers = []
   current_level_in = reshape(input)
   for hi,lo in zip(his,los):
      h = hi(current_level_in)
      l = lo(current_level_in)
      current_level_in = l
      out_layers.append(h)
   out_layers.append(l)

   analysis_layers_v = L.concatenate(out_layers, axis=1)

   # TODO add encoder layers: analysis → encoder → synthesis
   # L.Dense(units=encoder_size)
   #encoder_matrix = B.variable(np.array([[],[]]))
   #input_len *= 2
   encoder = L.Dense(units=input_len, weights=[np.eye(input_len), np.zeros(input_len)])
   decoder = L.Dense(units=input_len, weights=[np.eye(input_len), np.zeros(input_len)])

   #decoder_v = decoder(encoder(analysis_layers_v))
   #decoder_v = encoder(L.Flatten()(analysis_layers_v))
   decoder_v = L.Reshape([input_len])(analysis_layers_v)

   #synth_slices_v = [l(decoder_v) for l in synth_slices] 
   synth_slices_v = [l(analysis_layers_v) for l in synth_slices] 

   # --- collect all observables ---------------------------

   observables = [hi.output for hi in his]
   observables.append(los[-1].output)
   observables.append(analysis_layers_v)
   observables.append(decoder_v)
   observables.extend(synth_slices_v)

   model_f = B.function([input], observables)

   return model_f



model_f = build_dyadic_grid(2,input_len=8)
for x in model_f([ np.matrix([1,2,3,4,3,2,1,0]) ]):
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
