import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS

from timeshift_autoencoder import predictors as P


import keras.layers as L
from keras.layers import Input, Dense, Reshape, Activation, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras import losses
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import os


activation = fun.bind(XL.tanhx, alpha=0.1)


class AdversarialAutoencoder():

   def __init__(self):
      self.sig_len = 128
      self.img_shape = [self.sig_len]
      latent_sizes = [3, 3]
      self.z_size = sum(latent_sizes)

      optimizer = keras.optimizers.Adam(0.0002, 0.5)

      self.discriminator = self.make_discriminator()
      self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
      self.discriminator.summary()

      self.encoder, self.decoder, self.dec1, self.dec2 = self.make_autoencoder(self.sig_len, latent_sizes)

      #self.encoder = self.make_encoder()
      self.encoder.summary()

      #self.decoder, self.dec1, self.dec2 = self.make_decoder()
      self.decoder.compile(loss=['mse'], optimizer=optimizer)
      self.decoder.summary()

      img = Input(shape=self.img_shape)
      encoded_repr = self.encoder(img)
      reconstructed_img = self.decoder(encoded_repr)

      self.discriminator.trainable = False
      validity = self.discriminator(encoded_repr)

      self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
      self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
         loss_weights=[0.999, 0.001],
         optimizer=optimizer)


   @staticmethod
   def make_autoencoder(x_len, latent_sizes=[3, 3]):

      activation = fun.bind(XL.tanhx, alpha=0.2)
      act = lambda: L.Activation(activation)
      noise_stddev = 0.03
      eta = lambda: F.noise(noise_stddev)
      latent_size = sum(latent_sizes)

      encoder = (  F.dense([x_len//2])  >> act()                   # >> dropout(0.2)
                >> F.dense([x_len//4])  >> act() #>> batch_norm() # >> dropout(0.2)
                >> F.dense([latent_size]) >> act() #>> batch_norm() # >> dropout(0.2)
                )


      slice1 = XL.Slice[:,0:latent_sizes[0]]
      decoder1 = (  fun._ >> slice1
                 >> F.dense([x_len//4]) >> act() #>> batch_norm() # >> dropout(0.2)
                 >> F.dense([x_len//2]) >> act() #>> batch_norm() # >> dropout(0.2)
                 >> F.dense([x_len]) 
                 )

      slice2 = XL.Slice[:,latent_sizes[0]:]
      decoder2 = (  fun._ >> slice2
                 >> F.dense([x_len//4]) >> act() #>> batch_norm() # >> dropout(0.2)
                 >> F.dense([x_len//2]) >> act() #>> batch_norm() # >> dropout(0.2)
                 >> F.dense([x_len]) 
                 )

      x = L.Input([x_len])
      z = L.Input([latent_size])
      y1 = eta() >> encoder >> eta() >> decoder1
      y2 = eta() >> encoder >> eta() >> decoder2

      y = lambda x: L.Add()([y1(x), y2(x)])

      return (
         #M.Model([x], [y(x)]),
         M.Model([x], [(eta() >> encoder)(x)]),
         M.Model([z], [L.Add()([(eta() >> decoder1)(z), (eta() >> decoder2)(z)])]),
         M.Model([z], [decoder1(z)]),
         M.Model([z], [decoder2(z)]),
         #M.Model([x], [y1(x)]),
         #M.Model([x], [y2(x)]),
      )

   """
   @staticmethod
   def make_decoder(self):

      x = Input(shape=[self.z_size])
      y = XL.Slice[:,0:3](x)
      y = Dense(32)(y)
      y = Activation(activation)(y)
      y = BatchNormalization(momentum=0.8)(y)
      y = Dense(64)(y)
      y = Activation(activation)(y)
      y = BatchNormalization(momentum=0.8)(y)
      y = Dense(np.prod(self.img_shape))(y)
      y1 = y

      y = XL.Slice[:,3:6](x)
      y = Dense(32)(y)
      y = Activation(activation)(y)
      y = BatchNormalization(momentum=0.8)(y)
      y = Dense(64)(y)
      y = Activation(activation)(y)
      y = BatchNormalization(momentum=0.8)(y)
      y = Dense(np.prod(self.img_shape))(y)
      y2 = y

      return Model([x], [L.Add()([y1, y2])]), Model([x], [y1]), Model([x], [y2])
   """

   def make_discriminator(self):

      discriminator = Sequential()
      discriminator.add(Dense(64, input_dim=self.z_size))
      discriminator.add(Activation(activation))
      discriminator.add(BatchNormalization(momentum=0.8))
      discriminator.add(Dense(32))
      discriminator.add(Activation(activation))
      discriminator.add(BatchNormalization(momentum=0.8))
      discriminator.add(Dense(1, activation="sigmoid"))

      return discriminator


   def train(self, data, epochs, batch_size=128):

      half_batch = batch_size // 2

      valid = np.ones((half_batch, 1))
      fake = np.zeros((half_batch, 1))

      def random_batch(size):
         idx = np.random.randint(0, data.shape[0], size)
         return data[idx]

      def train_discriminator():
         imgs = random_batch(half_batch)

         latent_fake = self.encoder.predict(imgs)
         #latent_real = np.random.normal(size=(half_batch, self.z_size))
         latent_real = np.random.multivariate_normal(
            [0, 0, 0, 0, 0, 0],
            [ [1., .2, .2, 0., 0., 0.]
            , [.2, 1., .2, 0., 0., 0.]
            , [.2, .2, 1., 0., 0., 0.]
            , [0., 0., 0., 1., .2, .2]
            , [0., 0., 0., .2, 1., .2]
            , [0., 0., 0., .2, .2, 1.]
            ],
            size=(half_batch)
         )

         d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
         d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
         return 0.5 * np.add(d_loss_real, d_loss_fake)


      def train_generator():
         imgs = random_batch(batch_size)

         valid_y = np.ones((batch_size, 1))

         return self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid_y])


      for epoch in range(epochs):

         d_loss = train_discriminator()
         g_loss = train_generator()

         print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))



def make_prediction(model, frames, num=2000):
   frame_size = len(frames[0])
   pred_frames = model.predict(frames[:num])
   times = [np.arange(n, n+frame_size) for n in np.arange(len(pred_frames))]
   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, pred_frames):
      avg_frames[t] += f
   avg_frames *= 1./frame_size
   return avg_frames


def plot_joint_dist_impl(code, n_latent):
   fig, axs = plt.subplots(n_latent, n_latent, figsize=(8, 8))
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.5)
         ax.axis('off')

def plot_joint_dist(encoder, frames, n_latent):
   code = encoder.predict(frames)
   fig, axs = plt.subplots(n_latent, n_latent, figsize=(8, 8))
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.5)
         ax.axis('off')


if __name__ == '__main__':

    lorenz = lambda n: TS.lorenz(n, [1,0,0])[::1]
    fm_strong = lambda n: 0.5*np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
    sig_gen = lambda n: lorenz(n) + fm_strong(n)

    frames, *_ = TS.make_training_set(sig_gen, frame_size=128, n_pairs=6000, shift=1)

    aae = AdversarialAutoencoder()
    aae.train(frames, epochs=10000, batch_size=32)

    x = Input(shape=aae.img_shape)
    ae = Model([x], [aae.decoder(aae.encoder(x))])

    x = Input(shape=aae.img_shape)
    ae1 = Model([x], [aae.dec1(aae.encoder(x))])
    x = Input(shape=aae.img_shape)
    ae2 = Model([x], [aae.dec2(aae.encoder(x))])

    code = aae.encoder.predict(frames)
    TS.plot3d(*code.T[:3], 'k')

    plt.figure()
    plt.plot(make_prediction(ae1, frames, 2000))
    plt.plot(make_prediction(ae2, frames, 2000))


    def plot_joint_dist():
       code = aae.encoder.predict(frames)
       fig, axs = plt.subplots(aae.z_size, aae.z_size, figsize=(8, 8))
       for ax_rows, c1 in zip(axs, code.T):
          for ax, c2 in zip(ax_rows, code.T):
             ax.plot( c2, c1, '.k', markersize=0.5)
             ax.axis('off')

    plot_joint_dist()