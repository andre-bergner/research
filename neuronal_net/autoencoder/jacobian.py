import os
os.environ['KERAS_BACKEND'] = 'theano'

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
import numpy as np

import sys
sys.path.append('../')

from keras_tools import extra_layers as XL

jacobian = XL.jacobian


x = L.Input([4])
f = L.Dense(2, weights=[np.array([[1,2,3,4],[-4,-3,-2,-1]]).T], use_bias=False)
g = L.Dense(4, weights=[np.array([[1,2,3,4],[-4,-3,-2,-1]])], use_bias=False)

h = f(x)
y = g(h)

dhdx = jacobian(h,x)
dhdx_f = K.function([x],[dhdx])
print(dhdx_f([np.array([[1,2,2,1]])]))

m = M.Model([x], [y])

def contract(y_true, y_pred):
   return keras.losses.mean_squared_error(y_true, y_pred) + K.sum(dhdx*dhdx)

data = np.array([[1,1,0,0],[0,1,0,1]])
m.compile(optimizer='sgd', loss=contract)
m.fit(data,data)
print(m.predict(data))
