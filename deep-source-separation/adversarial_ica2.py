import numpy as np
import keras
import keras.layers as L
import keras.models as M
import keras.backend as K
import matplotlib.pyplot as plt

from keras_tools import extra_layers as XL
from result_tools import subplots_in
from test_data import *
from pylab import *



class AdversarialICA:

   def __init__(self,
      signals,
      adversarial=0.5,
      #optimizer=keras.optimizers.RMSprop(lr=0.001),
      optimizer=keras.optimizers.Adam(lr=0.001),
      discriminator_penalty=False,
      normalize_latent_space=False,
      latent_space_layout=None
   ):

      self.signals = signals
      num_channels = signals.shape[0]

      if latent_space_layout:
         assert(sum(latent_space_layout) == num_channels)
      else:
         latent_space_layout = np.ones(num_channels)

      latent_indices = np.concatenate([[0], np.cumsum(latent_space_layout)])
      self.latent_slices = np.array([latent_indices[:-1], latent_indices[1:]], dtype=int).T

      self.critic = M.Sequential([
         L.Dense(40, input_shape=(num_channels,)),
         L.LeakyReLU(0.1),
         L.Dense(20),
         L.LeakyReLU(0.1),
         L.Dense(1),
         L.Activation('sigmoid')
      ])

      if discriminator_penalty:
         grad = XL.jacobian(self.critic.output, self.critic.input)
         self.critic.add_loss(5*K.mean(K.square(grad)))
      self.critic.compile(loss='binary_crossentropy', optimizer=optimizer)
      # self.critic.compile(loss=lambda x, y: K.softplus(x * y), optimizer=optimizer)

      self.static_critic = M.Model(self.critic.input, self.critic.output)
      self.static_critic.trainable = False

      x = L.Input([num_channels])
      z = L.Input([num_channels])
      e = L.Dense(num_channels, use_bias=False, weights=[np.eye(num_channels)])
      # d = L.Dense(num_channels, use_bias=False)

      self.encoder = M.Model(x, e(x))
      # self.decoder = M.Model(z, d(z))
      # self.autoenc = M.Model(self.encoder.input, self.decoder(self.encoder.output))
      # self.autoenc.compile(loss='mse', optimizer=optimizer)

      # self.combined = M.Model(
      #    [self.autoenc.input],
      #    [self.autoenc.output, self.static_critic(self.encoder.output)]
      # )
      # self.combined.compile(
      #    loss=['mse', 'binary_crossentropy'],
      #    loss_weights=[1. - adversarial, adversarial],
      #    optimizer=optimizer
      # )

      # Classical GAN setup: the encoder (generator) directly feeding the critic
      self.criticized_encoder = M.Model([self.encoder.input], [self.static_critic(self.encoder.output)])
      if normalize_latent_space:
         self.criticized_encoder.add_loss(0.005*K.square(K.mean(self.encoder.outputs[0])))
         self.criticized_encoder.add_loss(0.005*K.square(1-K.mean(K.square(self.encoder.outputs[0]))))
      self.criticized_encoder.compile(loss=['binary_crossentropy'], optimizer=optimizer)

      self.g_losses = []
      self.d_losses = []
      self.weights = [self.encoder.get_weights()[0]]

      num_samples = self.signals.shape[1]
      self.idx = lambda n: np.random.randint(0, num_samples, n)


   def batch(self, batch_size):
      return self.signals[:, self.idx(batch_size)].T

   def pred(self, batch_size):
      return self.encoder.predict(self.batch(batch_size))
   
   def pred_perm(self, batch_size):
      batch = self.pred(batch_size)
      for s1, s2 in self.latent_slices:
         np.random.shuffle(batch[:, s1:s2])
      #for channel in batch.T: np.random.shuffle(channel)
      return batch

   @staticmethod
   def fake(n):
      return np.zeros(n)
      #return -np.ones(n)

   @staticmethod
   def real(n):
      return np.ones(n)


   def train(self, n_batches=1000, batch_size=256, discriminator_runs=1):
      for n in range(n_batches):
         batch = self.batch(batch_size)
         # _, g_loss, _ = self.combined.train_on_batch([batch], [batch, self.real(batch_size)])
         g_loss = self.criticized_encoder.train_on_batch([batch], [self.real(batch_size)])
         d_loss = self.train_critic(discriminator_runs, batch_size)

         self.g_losses.append(g_loss)
         self.d_losses.append(d_loss)
         self.weights.append(self.encoder.get_weights()[0])

         if n % 10 == 0:
            print( '\r', end='' )
            print( 'batch {}/{}  g_loss={:.3}  d_loss={:.3}'.format(
               n, n_batches, g_loss, d_loss), end='')
            print( '', end='', flush=True )

      print('')


   def train_critic(self, n_batches=1000, batch_size=128):
      loss = 0
      b_half = batch_size//2

      for n in range(n_batches):
         batch = np.concatenate([self.pred(b_half), self.pred_perm(b_half)])
         labels = np.concatenate([self.fake(b_half), self.real(b_half)])
         loss += self.critic.train_on_batch([batch], [labels])

      return loss / n_batches


   def pred_critic(self, batch_size=512):
      return np.array([
         self.critic.predict([self.pred(batch_size)]),
         self.critic.predict([self.pred_perm(batch_size)])
      ])



layout = None
num_samples = 10000

sig1, sig2, sig3 = fm_strong0, lorenz, make_sin_gen(0.1)
sig1, sig2 = make_sin_gen(np.pi*0.1), make_sin_gen(0.1)
signal = np.stack([
   (sig1 + 0.5*sig2)(num_samples),
   (0.33*sig1 - sig2)(num_samples),
#   (0.121*sig3 + sig1)(num_samples)
   (0.6*sig1 + 0.4*sig2)(num_samples),
   (-1.12*sig1 + 0.21*sig2)(num_samples),
])

layout = [2, 2]
w1 = np.pi*0.1
w2 = 0.1
sig1, sig2 = make_sin_gen(w1), make_sin_gen(w2)
mix = (sig1 + sig2)(num_samples + 3)
signal = np.stack([ mix[0:-4], mix[1:-3], mix[2:-2], mix[3:-1] ])  # embedding

#   the matrix to be learned:
mat = np.stack([
   np.cos( w1 * np.arange(4) ),
   np.sin( w1 * np.arange(4) ),
   np.cos( w2 * np.arange(4) ),
   np.sin( w2 * np.arange(4) ),
]).T
wat = np.linalg.inv(mat)
# plot( np.linalg.inv(M).dot(signal[:200]).T )


ica = AdversarialICA(signal, discriminator_penalty=True, latent_space_layout=layout)
ica.train(500, batch_size=512, discriminator_runs=20)


def plot_sep():
   figure()
   plot(ica.encoder.predict(signal.T))

def plot_dist(W=ica.encoder.get_weights()[-1]):
   figure()
   # y = ica.encoder.predict(signal.T)
   y = dot(signal.T, W)
   plot(*signal, '.', alpha=0.5, markersize=4)
   plot(*y.T, '.', alpha=0.5, markersize=4)


def subplots_in(n_rows, n_cols, fig, rect=[0,0,1,1]):

   xs = np.linspace(rect[0], rect[2], n_cols, endpoint=False)
   ys = np.linspace(rect[1], rect[3], n_rows, endpoint=False)
   width = (rect[2] - rect[0]) / (n_cols)
   hight = (rect[3] - rect[1]) / (n_rows)

   axs = [[fig.add_axes([x, y, width, hight]) for y in ys] for x in xs]
   return axs


def plot_summary(axes=[0,1]):
   fig = plt.figure(figsize=(8,8))

   ax = fig.add_axes([0.05, 0.7, 0.9, 0.22])
   plt.semilogy(ica.g_losses, label='separator')
   plt.semilogy(ica.d_losses, label='discriminator')
   plt.semilogy([0, len(ica.d_losses)], [np.log(2), np.log(2)], ':k', label=r'$\log 2$')
   plt.legend()
   fig.text(0.05, 0.94, 'losses', fontsize=16)

   num_plots = 8
   axs = subplots_in(1, num_plots, fig, [0.05,0.46,0.95,0.6])
   for ax, W in zip(axs, ica.weights[::len(ica.weights)//num_plots]):
      y = dot(signal.T, W)
      ax[0].plot(*y.T[axes], '.k', alpha=0.5, markersize=1)
      ax[0].axis('off')
   fig.text(0.05, 0.62, 'distribution evolution', fontsize=16)

   components = ica.encoder.predict(signal.T).T
   axs = subplots_in(components.shape[0], 1, fig, [0.05,0.05,0.95,0.4])
   for ax, s in zip(axs[0], components):
      ax.plot(s, 'k', linewidth=0.8)
      ax.axis('off')
   fig.text(0.05, 0.40, 'components', fontsize=16)


plot_summary()
