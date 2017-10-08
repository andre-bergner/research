import os
os.environ["KERAS_BACKEND"] = "theano"

import numpy as np
import keras
from keras import models as M
from keras import layers as L

import sys
sys.path.append('../')

from keras_tools import tools


data_size = 500
num_epochs = 300
categories = 2

def object_type(x,n):
   #return tools.gaussian(x, data_size,10*(n%categories+1))
   if n == 0:
      return tools.gaussian(x, data_size, 20)
   else:
      return tools.gaussian(x-15, data_size, 10) \
           + tools.gaussian(x+15, data_size, 10)


inputs = np.matrix([
   object_type(pos,n%categories)
   for pos,n in zip(np.random.rand(2500)*data_size, np.arange(2500))
])

validate = np.matrix([
   object_type(pos,n%categories)
   for pos,n in zip(np.random.rand(500)*data_size, np.arange(500))
])

features = np.matrix([
   tools.dirac(n%categories,categories) for n in np.arange(2500)
])

keras.backend.set_image_dim_ordering('th')

model = M.Sequential()
model.add(L.Dense(units=4, input_dim=data_size))
model.add(L.Dropout(0.3))
model.add(L.Activation('sigmoid'))
model.add(L.Dense(units=categories))
model.add(L.Activation('softmax'))

'''
model = M.Sequential()
model.add(L.Reshape((data_size,1), input_shape=(data_size,)))
model.add(L.Conv1D(4, kernel_size=(10)))
model.add(L.Dropout(0.3))
model.add(L.Activation('sigmoid'))
model.add(L.Conv1D(4, kernel_size=(10)))
model.add(L.Flatten())
model.add(L.Dropout(0.3))
model.add(L.Activation('sigmoid'))
model.add(L.Dense(units=categories, input_dim=data_size))
model.add(L.Activation('softmax'))
'''

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1.0))
model.fit( inputs, features, epochs=num_epochs, batch_size=30
         , verbose=0, callbacks=[tools.Logger()] )

tools.validate_model(model, inputs, features)
tools.validate_model(model, validate, features[:500])


import pylab as pl

#pl.imshow( model.layers[0].get_weights()[0], aspect='auto', cmap='gray')
pl.plot( model.layers[0].get_weights()[0])
