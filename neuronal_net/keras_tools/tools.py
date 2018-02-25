import keras.callbacks
import numpy as np
import pylab as pl


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

   def __init__(self, **kargs):
      super(LossRecorder, self).__init__(**kargs)
      self.losses = []
      self.grads = []

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights




class Logger(keras.callbacks.Callback):

   def __init__(self):
      self.bar_size = 30

   def on_epoch_end(self, epoch, logs={}):
      num_epochs = self.params['epochs']
      filled_bars = int(epoch * self.bar_size / num_epochs) + 1
      nonfilled_bars = self.bar_size - filled_bars
      print( '\r', end='' )
      print( 'training: [' + filled_bars*'●' + nonfilled_bars*' ' + ']  ', end='' )
      print( 'Epoch {0}/{1} (loss={2:.3})'. format(epoch+1, num_epochs, logs.get('loss')), end='')
      print( '', end='', flush=True )

   def on_train_end(self, logs={}):
      print(' ✔')


def train(model, input, output, batch_size, epochs, loss_recorder=None):
   if loss_recorder == None:
      loss_recorder = LossRecorder()
   model.fit(
      input, output, batch_size=batch_size, epochs=epochs, verbose=0,
      callbacks=[Logger(), loss_recorder])
   return loss_recorder


def observe_all_layers(model):
   def all_outputs(layer):
      return [layer.get_output_at(n) for n in range(len(layer.inbound_nodes))]
   layer_outputs = [all_outputs(l) for l in model.layers[1:]]
   return K.function(
      [model.layers[0].input],
      [l for ll in layer_outputs for l in ll]
   )


class CascadeFactory:

   def __init__(self, factory, shared=True):
      if shared:
         artefact = factory();
         self.factory = lambda: artefact
      else:
         self.factory = factory

   def get(self):
      return self.factory()


def print_layer_outputs(model):
   for l in model.layers:
      print("{:>20} : {}".format(l.name, l.output_shape[1:]))


def plot_target_vs_prediction(model, inputs, targets, n=0, ax=None):
   if ax == None:
      pl.figure()
      plot = pl.plot
   else:
      plot = ax.plot
   plot(targets[n], 'k')
   plot(model.predict(inputs[n:n+1])[0], 'r')


def plot_top_and_worst(model, inputs, targets, num=3):
   dist = [
      (model.evaluate(inputs[n:n+1],targets[n:n+1],verbose=0), n)
      for n in range(len(inputs))
   ]
   dist.sort()
   fig, ax = pl.subplots(num,2)
   for n in range(num):
      plot_target_vs_prediction(model, inputs, targets, dist[n][1], ax[n,0])
      plot_target_vs_prediction(model, inputs, targets, dist[-n-1][1], ax[n,1])


def add_noise(data, sigma=0.0):
   return data + sigma * np.random.randn(*data.shape)
