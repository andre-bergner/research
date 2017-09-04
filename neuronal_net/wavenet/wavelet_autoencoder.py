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


def make_analysis_node(kernel):
   return L.Conv1D(1, kernel_size=(len(kernel)), strides=(2), weights=[np.array([[kernel]]).T, np.zeros(1)])

def make_lo(): return make_analysis_node([1,1])
def make_hi(): return make_analysis_node([1,-1])


def build_dyadic_grid(num_levels=3, encoder_size=10, input_len=None):

   # --- create all operators (nodes) ----------------------
   input_len = input_len or 2**num_levels
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))
   analysis_filters = []
   synth_slices = []     # the slices for the synthesis network
   synth_filters = []
   right_crop = input_len # 2**num_levels
   left_crop = 0
   for _ in range(num_levels):
      analysis_filters.append(( make_hi(), make_lo() ))
      synth_filters.append(( make_hi(), make_lo() ))
      right_crop >>= 1
      synth_slices.append(L.Cropping1D([left_crop, right_crop]))
      left_crop += right_crop
   flatten = L.Flatten()
   synth_slices.append(L.Cropping1D([left_crop, 0]))

   # TODO: wavelet weights must be tied!

   # --- connect nodes to graph ----------------------------
   out_layers = []
   current_level_in = reshape(input)
   for hi,lo in analysis_filters:
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
   reshape2 = L.Reshape((input_len,1))

   decoder_v = reshape(decoder(encoder(L.Flatten()(analysis_layers_v))))

   synth_slices_v = [l(decoder_v) for l in synth_slices]

   # for hi,lo in magic_zip( synth_filters, synth_slices_v ):
   #    L.UpSampling1D(2)()

   # --- collect all observables ---------------------------

   observables = [hi.output for hi,_ in analysis_filters]
   observables.append(analysis_filters[-1][1].output)
   #observables.append(analysis_layers_v)
   #observables.append(decoder_v)
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
