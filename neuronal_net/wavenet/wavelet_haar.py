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

# create operator nodes
input = L.Input(shape=(4,))
reshape = L.Reshape((4,1))
hi1 = make_hi()
lo1 = make_lo()
hi2 = make_hi()
lo2 = make_lo()
flatten = L.Flatten()

# connect noes to graph
in3d = reshape(input)
h1 = hi1(in3d)
l1 = lo1(in3d)
h2 = hi2(l1)
l2 = lo2(l1)
#model_out = L.concatenate([L.Flatten()(h1),L.Flatten()(h2),L.Flatten()(l2)])
model_out = L.concatenate([h1,h2,l2],axis=1)
# current output is in pyramids
# → h1 needs to be tranposed before flatten, options (both not working) .T or B.transpose()


model1_f = B.function([input], [hi1.output, hi2.output, lo2.output, model_out])
print("Using Strides")

hi1_o, hi2_o, lo2_o, net_o = model1_f([ np.matrix([1,2,3,4,3,2,1,0]) ])

print( "hi1: {}".format(hi1_o.flatten()) )
print( "hi2: {}".format(hi2_o.flatten()) )
print( "lo2: {}".format(lo2_o.flatten()) )
#print( "all: {}".format(net_o.T.flatten()) )
print( "pyramids:\n{}".format(net_o[:,:,0]) )

