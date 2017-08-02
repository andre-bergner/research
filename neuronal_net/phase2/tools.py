import keras.callbacks
import numpy as np


def gaussian(pos, size=1024, sigma=40):
   return np.exp( -((np.arange(size) - pos) / sigma)**2 )


def dirac(pos, size=1024):
   d = np.zeros(int(size))
   d[int(pos)] = 1.
   return d




def validate_model(model, inputs, features):

   correct_results = [
      np.argmax(num) == np.argmax(model.predict(img))
      for img, num in zip(inputs,features)
   ]

   print(
      "correct results: {0:.2f} %"
      .format(
         100. * float(np.count_nonzero(correct_results)) / len(correct_results)
      )
   )


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




class Logger(keras.callbacks.Callback):

   def __init__(self, num_epochs):
      self.num_epochs = num_epochs
      self.bar_size = 30

   def on_epoch_begin(self, epoch, logs={}):
      #print( "\rEpoch {0}/{1}, {2:.0f}%   "
      #     . format(j, n_epochs, 100.*float(n_batch)/len(mini_batches))
      #     , end="", flush=True)
      filled_bars = int(epoch * self.bar_size / self.num_epochs) + 1
      nonfilled_bars = self.bar_size - filled_bars
      print( '\r', end='' )
      print( 'training: [' + filled_bars*'●' + nonfilled_bars*' ' + ']  ', end='' )
      print( 'Epoch {0}/{1}'. format(epoch+1, self.num_epochs), end='')
      print( '', end='', flush=True )

   def on_train_end(self, logs={}):
      print(' ✔')

   