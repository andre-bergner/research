import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

import sys
sys.path.append('../')

from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS
from keras_tools.upsampling import UpSampling1DZeros, DownSampling1D



# --------------------------------------------------------------------------------------------------
# ENCODER/DECODER FACTORIES
# --------------------------------------------------------------------------------------------------


activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)
#act = lambda: L.LeakyReLU(alpha=0.2)  # does work too, but ugly latent space


class DenseFactory:

   def __init__(self, example_frame, latent_sizes):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])

   def make_encoder(self):
      latent_size = sum(self.latent_sizes)

      return (  F.dense([self.input_size//2])  >> act()                   # >> F.dropout(0.2)
             >> F.dense([self.input_size//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> F.dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.1)
             )

   def make_decoder(self):
      return (  F.dense([self.input_size//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> F.dense([self.input_size//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> F.dense([self.input_size])
             )



class ConvFactory:

   def __init__(self, example_frame, latent_sizes, kernel_size=5, features=[4, 4, 8, 8, 16, 16, 16],
      use_batch_norm=False, scale_factor=1, resnet=False
   ):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.kernel_size = kernel_size
      self.features = features
      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x
      self.scale_factor = scale_factor

      if resnet:
         self.enc_block = self.enc_residual_block
         self.dec_block = self.dec_residual_block
      else:
         self.enc_block = self.enc_normal_block
         self.dec_block = self.dec_normal_block


   @staticmethod
   def up1d(factor=2):
      #return fun._ >> UpSampling1DZeros(factor)
      return fun._ >> L.UpSampling1D(factor)

   def enc_residual_block(self, n_features_in, n_features_out):
      def block_f(x):
         #block = F.conv1d(n_features_out, self.kernel_size) >> DownSampling1D(2) >> act() >> self.batch_norm()
         block = F.conv1d(n_features_out, self.kernel_size) >> act() >> F.conv1d(n_features_out, self.kernel_size, 2)
         blockx = block(x)
         if n_features_out != n_features_in:
            skip_connection = F.flatten() >> F.dense(blockx._keras_shape[1:])
         else:
            skip_connection = DownSampling1D(2)
         # if n_features_out < n_features_in:
         #    skip_connection = K.stack
         # elif n_features_out > n_features_in:
         #    skip_connection = skip_connection[:,:,:n_features_out]
         return L.add([blockx, skip_connection(x)])
      return fun._ >> block_f >> act()

   def enc_normal_block(self, n_features_in, n_features_out):
      return F.conv1d(n_features_out, self.kernel_size, 2) >> act() >> self.batch_norm() # >> F.noise(0.1)

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      block = self.enc_block
      return (  F.append_dimension()
             >> block(1,           features[0])
             >> block(features[0], features[1])
             >> block(features[1], features[2])
             >> block(features[2], features[3])
             >> block(features[3], features[4])
             >> block(features[4], features[5])
             >> block(features[5], features[6])
             #>> block(features[6], features[7])
             >> F.flatten()
             >> F.dense([latent_size]) >> act()
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.01, no_sampling=True)
             )


   def dec_residual_block(self, n_features_in, n_features_out):
      def block_f(x):
         up = self.up1d
         block = up(2) >> F.conv1d(n_features_out, self.kernel_size) >> act() >> F.conv1d(n_features_out, self.kernel_size)
         blockx = block(x)
         if n_features_out != n_features_in:
            skip_connection = F.flatten() >> F.dense(blockx._keras_shape[1:])
         else:
            skip_connection = up(2)
         return L.add([blockx, skip_connection(x)])
      #return self.up1d() >> block_f >> act()
      return fun._ >> block_f >> act()

   def dec_normal_block(self, _, n_features_out):
      return self.up1d() >> F.conv1d(n_features_out, self.kernel_size) >> act() >> self.batch_norm() #>> F.noise(0.1)


   def make_decoder(self):
      up = self.up1d
      features = self.features
      block = self.dec_block
      return (  F.dense([self.scale_factor, features[-1]]) >> act() >> self.batch_norm() # >> F.noise(0.5)
             >> block(features[-1], features[-2])
             >> block(features[-2], features[-3])
             >> block(features[-3], features[-4])
             >> block(features[-4], features[-5])
             >> block(features[-5], features[-6])
             >> block(features[-6], features[-7])
             #>> block(features[-7], features[-8])
             >> up() >> F.conv1d(1, self.kernel_size)
             >> F.flatten()
             )




class ConvFactory2:

   def __init__(self, example_frame, latent_sizes, kernel_size=5, features=[4, 4, 8, 8, 16, 16, 16],
      use_batch_norm=False, scale_factor=1
   ):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.kernel_size = kernel_size
      self.features = features
      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x
      self.scale_factor = scale_factor

      # 156
      def layer_sizes(n_input, n_kernel):
         layer_size = n_input
         layer_sizes = [layer_size]
         while layer_size >= n_kernel:
            layer_size = (layer_size - (n_kernel-1)) // 2
            layer_sizes.append(layer_size)
         return layer_sizes


   @staticmethod
   def up1d(factor=2):
      return fun._ >> UpSampling1DZeros(factor)

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size


      return (  F.append_dimension()
             >> F.conv1d(features[0], ks, 2, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], ks, 1, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], 1,  2, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], ks, 1, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], 1,  2, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], ks, 1, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], 1,  2, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], ks, 1, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], 1,  2, padding='valid') >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> F.dense([latent_size]) >> act()
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.01, no_sampling=True)
             )

   def make_decoder(self):
      up = self.up1d
      features = self.features
      ks = self.kernel_size
      return (  F.dense([self.kernel_size*2+2, features[3]]) >> act() >> self.batch_norm()
             #>> up(self.kernel_size*2+2)
             #>> up() >> F.conv1d(features[3], ks, padding='valid') >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[2], ks, padding='valid') >> act() >> self.batch_norm()
                     >> F.conv1d(features[2], 1, padding='valid')  >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[1], ks, padding='valid') >> act() >> self.batch_norm()
                     >> F.conv1d(features[1], 1, padding='valid')  >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[0], ks, padding='valid') >> act() >> self.batch_norm()
                     >> F.conv1d(features[0], 1, padding='valid')  >> act() >> self.batch_norm()
             >> up() >> F.conv1d(1, ks, padding='valid')
             >> F.flatten()
             )


class DilatedFactory:

   def __init__(self, example_frame, latent_sizes, kernel_size=5, use_batch_norm=False, scale_factor=1):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.kernel_size = kernel_size
      self.features = 8
      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x
      self.scale_factor = scale_factor

   @staticmethod
   def up1d(factor=2):
      return fun._ >> L.UpSampling1D(factor)

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      return (  F.append_dimension()
             >> F.conv1d(features, ks, dilate=1) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=2) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=4) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=8) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=16) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=32) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=64) >> act() >> self.batch_norm()
             >> F.conv1d(latent_size, 128, padding='valid') >> act() >> self.batch_norm()
             >> F.flatten()
             )

   def make_decoder(self):
      up = self.up1d
      features = self.features
      ks = self.kernel_size
      return (  F.append_dimension(axis=-2)
             #fun._ >> L.ZeroPadding1D([0, self.input_size-1])
             >> F.conv1d(features, 1, dilate=64) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=64) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=32) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=16) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=8) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=4) >> act() >> self.batch_norm()
             >> F.conv1d(features, ks, dilate=2) >> act() >> self.batch_norm()
             >> F.conv1d(1, ks) >> act() >> self.batch_norm()
             >> F.flatten()
             )

# --------------------------------------------------------------------------------------------------
# FACTOR MODEL -- actual source separator that factors the state/latent space
# --------------------------------------------------------------------------------------------------


def rotate(xs):
   yield xs[-1]
   for x in xs[:-1]:
      yield x


def make_factor_model(
   example_frame, factory, noise_stddev, noise_decay=0, final_stddev=0,
   shared_encoder=True, vanishing_xprojection=False
):

   x = F.input_like(example_frame)
   #x_2 = F.input_like(example_frame)    # The Variational layer causes conflicts if this is in and not connected
   eta = lambda: F.noise(noise_stddev, decay=noise_decay, final_stddev=final_stddev)
   #eta2 = lambda: F.noise(0.1)


   if shared_encoder:
      encoder = factory.make_encoder()
      cum_latent_sizes = np.concatenate([[0], np.cumsum(factory.latent_sizes)])
      decoders = [
         fun._ >> XL.Slice[:, cum_latent_sizes[n]:cum_latent_sizes[n+1]] >> factory.make_decoder()
         # for DILATED: fun._ >> XL.Slice[:,:, cum_latent_sizes[n]:cum_latent_sizes[n+1]] >> factory.make_decoder()
         for n in range(len(factory.latent_sizes))
      ]
      ex = (eta() >> encoder)(x)
      channels = [dec(ex) for dec in decoders]
   else:
      encoders = [factory.make_encoder(z) for z in factory.latent_sizes]
      decoders = [factory.make_decoder() for n in range(len(factory.latent_sizes))]
      channels = [(eta() >> enc >> dec)(x) for enc, dec in zip(encoders, decoders)]
      ex = L.concatenate([e(x) for e in encoders])

   #channels = [(dec >> eta2() >> encoder >> dec)(ex) for dec in decoders]

   y = L.add(channels)

   m = M.Model([x], [y])
   #m.add_loss(10*K.mean( K.square(ex[:,0:2])) * K.mean(K.square(ex[:,2:])))

   #ex_2 = (eta() >> encoder)(x_2)
   #m_slow_feat = M.Model([x, x_2], [y])
   #m_slow_feat.add_loss(1*K.mean(K.square( ex_2[:,0:2] - ex[:,0:2] + ex_2[:,4:6] - ex[:,4:6] )))

   if vanishing_xprojection:
      # for c1, c2 in zip(channels, rotate(channels)):
      #    m.add_loss(1*K.mean(K.square(c1(c2))))
      for d,c in zip(decoders, rotate(channels)):
         m.add_loss(1*K.mean(K.square((encoder >> d)(c))))

   return (
      m,
      M.Model([x], [ex]),
      #M.Model([x], [(encoder >> F.flatten())(x)]),
      None,#m_slow_feat,
      [M.Model([x], [c]) for c in channels]
   )
