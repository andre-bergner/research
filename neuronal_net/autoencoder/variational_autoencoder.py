import sys
sys.path.append('../')

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.utils import plot_model

from keras_tools import functional_layers as F
from keras_tools import tools


class VariationalEncoder(L.Layer):

   def __init__(self, latent_size, data_size, *args, **kwargs):
      super(VariationalEncoder, self).__init__(*args, **kwargs)
      self.latent_size = latent_size
      self.data_size = data_size

   def compute_output_shape(self, input_shape):
      return (input_shape[0], self.latent_size)

   def call(self, inputs, training=None):

      def reparameterization(args):
         mu, log_sigma = args
         epsilon = K.random_normal(shape=K.shape(mu))
         sigma = K.exp(0.5 * log_sigma)
         return K.in_train_phase(mu + sigma * epsilon, mu + sigma, training=training)
         #return mu + sigma * epsilon

      h = inputs
      mu = F.dense([self.latent_size], name='mu')(h)
      log_sigma = F.dense([self.latent_size], name='log_sigma')(h)
      z = L.Lambda(reparameterization, output_shape=(self.latent_size,), name='z')([mu, log_sigma])

      kl_div = -.5 * K.mean(1 + log_sigma - K.square(mu) - K.exp(log_sigma))
      self.add_loss(kl_div * self.latent_size / self.data_size)

      self.mu = mu
      self.log_sigma = log_sigma

      return z




width, height = 28, 28
x_dim = width * height
z_dim = 20
act = lambda a: L.Activation(a)

x = L.Input([width, height], name='x')
encoder = F.flatten() >> F.dense([160]) >> act('relu') >> VariationalEncoder(z_dim, x_dim)
z = encoder(x)

decoder = (  F.dense([160]) >> act('relu')
          >> F.dense([width, height]) >> act('sigmoid')
          )

model = M.Model([x], [decoder(z)])

model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

# plot_model(model, to_file='vae.png', show_shapes=True)

from keras.datasets import mnist
(data, _), (_, _) = mnist.load_data()
data = data/255.

loss_recorder = tools.LossRecorder()
tools.train(model, data, data, 32, 5, loss_recorder)



from pylab import *

def ims(n):
   img = model.predict(data[n:n+1])[0]
   fig, axs = subplots(1,2, figsize=(4,2))
   axs[0].imshow(data[n], cmap='gray')
   axs[1].imshow(img, cmap='gray')

ims(1)
ims(2)
ims(3)
