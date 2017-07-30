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

keras.backend.set_image_dim_ordering('th')

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


def make_dense_model():

   model = kemod.Sequential()
   model.add(kelay.Dense(units=30, input_dim=784))
   model.add(kelay.Activation('sigmoid'))
   model.add(kelay.Dense(units=10))
   model.add(kelay.Activation('sigmoid'))

   return model


def make_conv_model():

   model = kemod.Sequential()
   model.add(kelay.Reshape((1,28,28), input_shape=(784,)))
   model.add(kelay.Conv2D(16, kernel_size=(5,5)))
   model.add(kelay.MaxPooling2D(pool_size=(2,2)))
   model.add(kelay.Dropout(0.3))
   model.add(kelay.Activation('relu'))
   model.add(kelay.Conv2D(10, kernel_size=(3,3)))
   model.add(kelay.Dropout(0.4))
   model.add(kelay.Activation('relu'))
   model.add(kelay.Flatten())
   model.add(kelay.Dense(10))
   model.add(kelay.Activation('softmax'))

   return model


#model = make_dense_model()
model = make_conv_model()


#model.compile(loss='categorical_crossentropy', optimizer='sgd')
#model.compile(loss='mean_squared_error', optimizer='sgd')
#model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.5,momentum=0.2,decay=0.01))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1.0))
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

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


import mnist_gui

class Predictor:

   def __init__(self, img, bar_ax, txt):
      self.img = img
      self.bar_ax = bar_ax
      self.txt = txt

   def predict(self):
      distribution = model.predict(matrix(self.img.array.flatten()))
      most_likely = np.argmax(distribution)
      self.bar_ax.clear()
      self.bar_ax.bar(arange(10), distribution.T)
      self.bar_ax.xaxis.set_ticks(np.arange(10)+.4)
      self.bar_ax.xaxis.set_ticklabels(np.arange(10))
      self.txt.set_text(most_likely)
      # print(most_likely)

predictor = Predictor(mnist_gui.draw_img, mnist_gui.bar_ax, mnist_gui.txt)
mnist_gui.btn_predict.on_clicked(lambda _: predictor.predict())

