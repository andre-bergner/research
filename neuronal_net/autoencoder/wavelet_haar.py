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

def build_dyadic_grid(num_levels = 3):

   # --- create all operators (nodes) ----------------------
   input_len = 2**num_levels
   input = L.Input(shape=(input_len,))
   reshape = L.Reshape((input_len,1))
   his = []
   los = []
   for _ in range(num_levels):
      his.append(make_hi())
      los.append(make_lo())
   flatten = L.Flatten()

   # --- connect nodes to graph ----------------------------
   out_layers = []
   current_level_in = reshape(input)
   for hi,lo in zip(his,los):
      h = hi(current_level_in)
      l = lo(current_level_in)
      current_level_in = l
      out_layers.append(h)
   out_layers.append(l)

   model_out = L.concatenate(out_layers, axis=1)
   # current output is in pyramids (stack of layers)
   # → h1 needs to be tranposed before flatten, options (both not working) .T or B.transpose()

   observables = [hi.output for hi in his]
   observables.append(los[-1].output)
   observables.append(model_out)

   model_f = B.function([input], observables)

   return model_f


model1_f = build_dyadic_grid(2)
print("Using Strides")

hi1_o, hi2_o, lo2_o, net_o = model1_f([ np.matrix([1,2,3,4,3,2,1,0]) ])

print( "hi1: {}".format(hi1_o.flatten()) )
print( "hi2: {}".format(hi2_o.flatten()) )
print( "lo2: {}".format(lo2_o.flatten()) )
#print( "all: {}".format(net_o.T.flatten()) )
print( "pyramids:\n{}".format(net_o[:,:,0]) )

