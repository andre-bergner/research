import keras


class Metrics(keras.callbacks.Callback):

   def __init__(self, predictor, **kargs):
      super(Metrics, self).__init__(**kargs)
      self.losses = []
      self.grads = []
      self.predictor = predictor
      self.predictions = []

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights

   # def on_epoch_end(self, batch, logs={}):
   #  TODO
   #  • measure prediction range
   #  • measure signal similarity (histogram, stroboscopic map, ?)

   def on_train_end(self, logs={}):
      self.predictions.append(self.predictor())
