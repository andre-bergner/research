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
   def on_train_begin(self, logs={}):
      self.losses = []

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))

loss_recorder = LossRecorder()

with timer.Timer() as t:
   model.fit( training_data[0], training_data[1]
            , epochs = n_epochs
            , batch_size = mini_batch_size
            , callbacks = [loss_recorder]
            )

"""
def minimize_loss(input, expected, eta=.5):
   loss = model.train_on_batch(input, expected)
   return np.asscalar(loss)

with timer.Timer() as t:
   losses = sgd.stochastic_gradient_descent(minimize_loss, training_data, n_epochs, mini_batch_size)
"""


correct_results = [ np.argmax(num) == np.argmax(model.predict(img)) for img, num in zip(mnist_valid[0],mnist_valid[1]) ]
#correct_results = [ np.argmax(num) == np.argmax(model.predict(img)) for img, num in mnist_valid ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )

losses = np.array(loss_recorder.losses)
plot(losses)
