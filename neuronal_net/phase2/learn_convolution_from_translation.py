import os
os.environ["KERAS_BACKEND"] = "theano"

import numpy as np
import keras
from keras import models as M
from keras import layers as L
import tools


data_size = 300
num_epochs = 500
downsampling = 10

nums = np.linspace(0,data_size-1,300)
inputs = np.matrix([ tools.gaussian(n,data_size) for n in nums ])
features = np.matrix([ tools.dirac(n/downsampling, data_size/downsampling) for n in nums ])


model = M.Sequential()
model.add(L.Dense(units=30, input_dim=data_size))
model.add(L.Activation('sigmoid'))


model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1.0))
model.fit( inputs, features, epochs=num_epochs, batch_size=20
         , verbose=0, callbacks=[tools.Logger(num_epochs)] )

tools.validate_model(model, inputs, features)


import pylab as pl

pl.imshow( model.layers[0].get_weights()[0], aspect='auto', cmap='gray')
