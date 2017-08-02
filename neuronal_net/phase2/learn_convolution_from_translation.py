# TODO
# * non-verbose model.fit or better: one global progress incl all epochs
# * document insight: number of features must be small
# * next steps:
#   * learn postion for diff objects
#   * learn diff objects at diff positions
#      -->  two layers , 3 layers: stacked or 2 parallel + 1 final


import os
os.environ["KERAS_BACKEND"] = "theano"

import numpy as np
import keras
from keras import models as M
from keras import layers as L


data_size = 300
num_epochs = 500

def gaussian(pos,sigma=40,size=data_size):
   return np.exp( -((np.arange(size) - pos) / sigma)**2 )

def dirac(n,s=data_size):
   d = np.zeros(s)
   d[int(n)] = 1.
   return d

inputs = np.matrix([ gaussian(n) for n in np.linspace(0,data_size-1,300) ])
features = np.matrix([ dirac(n/10,30) for n in np.linspace(0,data_size-1,300) ])


model = M.Sequential()
model.add(L.Dense(units=30, input_dim=data_size))
model.add(L.Activation('sigmoid'))


model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1.0))
model.fit(inputs, features, epochs=num_epochs, batch_size=20)

correct_results = [ np.argmax(num) == np.argmax(model.predict(img)) for img, num in zip(inputs,features) ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

import pylab as pl

pl.imshow( model.layers[0].get_weights()[0], aspect='auto', cmap='gray')
