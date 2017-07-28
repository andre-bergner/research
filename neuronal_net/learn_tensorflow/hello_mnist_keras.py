import os
os.environ["KERAS_BACKEND"] = "theano"

import keras
import keras.models as kemod
import keras.layers as kelay

import numpy as np
import timer
import sgd
from pylab import *

n_epochs        = 3
mini_batch_size = 20
eta             = 0.1

def farray(a):
   return np.array(a,dtype=np.float32)

def load_mnist( path = "mnist.pkl.gz" ):

   import gzip
   import pickle

   with gzip.open(path, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
   
   return data  # training_data, validation_data, test_data

def reshape_mnist_data( data ):
   def dirac(n):
      d = np.zeros(10); d[n]=1; return d;
   #return [ (farray(x),farray(dirac(n))) for x,n in zip(data[0],data[1]) ]
   return np.matrix(data[0]), np.matrix([ dirac(n) for n in data[1] ])
   #return [ (np.matrix(x),np.matrix(dirac(n))) for x,n in zip(data[0],data[1]) ]


# build network

model = kemod.Sequential()
model.add(kelay.Dense(units=30, input_dim=784))
model.add(kelay.Activation('sigmoid'))
model.add(kelay.Dense(units=10))
model.add(kelay.Activation('sigmoid'))

#model.compile(loss='categorical_crossentropy', optimizer='sgd')
#model.compile(loss='mean_squared_error', optimizer='sgd')
#model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.5,momentum=0.2,decay=0.01))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1.0))

# load data

training_data, mnist_valid, test_data = load_mnist()
training_data = reshape_mnist_data(training_data)
mnist_valid = reshape_mnist_data(mnist_valid)
test_data = reshape_mnist_data(test_data)


class LossRecorder(keras.callbacks.Callback):
   def __init__(self, model):
      self.model = model

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.losses = []
      self.grads = []
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights

loss_recorder = LossRecorder(model)

with timer.Timer() as t:
   model.fit( training_data[0], training_data[1]
            , epochs = n_epochs
            , batch_size = mini_batch_size
            , callbacks = [loss_recorder]
            )


correct_results = [ np.argmax(num) == np.argmax(model.predict(img)) for img, num in zip(mnist_valid[0],mnist_valid[1]) ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )



figure()

subplot(211)
title("Losses")
semilogy(np.array(loss_recorder.losses),'k')

subplot(212)
title("∆weights")
semilogy(abs(np.array(loss_recorder.grads)))


