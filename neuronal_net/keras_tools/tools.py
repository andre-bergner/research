import keras.callbacks
import numpy as np
import pylab as pl
import time


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

   def on_train_begin(self, logs={}):
      self.start_time = time.time()
      self.num_epochs = self.params['epochs']
      self.batches_per_epoch = self.params['samples'] / self.params['batch_size']
      self.all_batches = self.num_epochs * self.batches_per_epoch
      self.current_batch = 0

   def on_epoch_begin(self, epoch, logs={}):
      self.current_epoch = epoch

   def on_batch_begin(self, batch, logs={}):
      self.current_batch += 1

   def on_batch_end(self, batch, logs={}):
      filled_bars = int(self.current_batch * self.bar_size / self.all_batches)
      nonfilled_bars = self.bar_size - filled_bars
      delta_time = time.time() - self.start_time
      print( '\r', end='' )
      print( 'training: [' + filled_bars*'=' + nonfilled_bars*' ' + ']  ', end='' )
      print( 'Epoch {0}/{1} (loss={2:.3}, {3:.1f} sec)'.format(
         self.current_epoch+1, self.num_epochs, logs.get('loss'), delta_time), end='')
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

def visulize_graph(model, filename):
   if keras.backend.backend() == 'theano':
      from theano.printing import pydotprint
      pydotprint(model.outputs[0], filename)
      #import theano.d3viz as d3v
      #d3v.d3viz(model.outputs[0], filename)



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
