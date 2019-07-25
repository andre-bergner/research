from functools import reduce
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


soft_relu = lambda alpha=0.2: fun.bind(XL.soft_relu, alpha=alpha)
leaky_tanh = lambda alpha=0.2: fun.bind(XL.tanhx, alpha=alpha)
activation = leaky_tanh(0.2)
act = lambda: L.Activation(activation)
#act = lambda: L.LeakyReLU(alpha=0.2)  # does work too, but ugly latent space
identity = lambda x: x


class DenseFactory:

   def __init__(self, input_size, latent_sizes, layer_sizes=None, decoder_noise=None, vae=False):
      self.input_size = input_size
      self.latent_sizes = latent_sizes
      self.layer_sizes = layer_sizes if layer_sizes else [input_size//2, input_size//4]

      if isinstance(decoder_noise, (int, float)):
         self.dec_noise = lambda: F.noise(decoder_noise)
      elif isinstance(decoder_noise, dict):
         self.dec_noise = lambda: F.noise(**decoder_noise)
      else:
         self.dec_noise = lambda: identity

      latent_size = sum(self.latent_sizes)
      self.latent_layer = \
         XL.VariationalEncoder(latent_size, self.input_size, beta=0.1) if vae else \
         F.dense([latent_size]) >> act()

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      layers = reduce(
         lambda x, y: x >> act() >> y,
         [F.dense([s]) for s in self.layer_sizes]
      )
      return layers >> act() >> self.latent_layer

   def make_decoder(self, custom_features=None, custom_noise=None):
      # TODO implement custom features, layers (if it makes sense)
      layers = reduce(
         lambda x, y: x >> y,
         [F.dense([s]) >> act() >> self.dec_noise() for s in self.layer_sizes[::-1]]
      )
      return layers >> F.dense([self.input_size])



class ConvFactory:

   def __init__(self,
      input_size,
      latent_sizes,
      kernel_size=5,
      final_kernel_size=None,
      upsample_with_zeros=False,   # TODO this should be True but is false for backwards compatibility
      features=[4, 4, 8, 8, 16, 16, 16],
      dec_features=None,
      use_batch_norm=False,
      resnet=False,
      skip_connection=False,
      one_one_conv=False,
      decoder_noise=None,
      decoder_kernel_regularizer=None,
      activation=leaky_tanh(0.2)
   ):
      self.input_size = input_size
      self.latent_sizes = latent_sizes
      self.kernel_size = kernel_size
      self.final_kernel_size = final_kernel_size or kernel_size
      self.upsample_with_zeros = upsample_with_zeros
      self.features = features
      self.dec_features = dec_features if dec_features else features[::-1]
      self.one_one_conv = one_one_conv
      self.act = lambda name=None: L.Activation(activation, name=name)
      self.decoder_kernel_regularizer = decoder_kernel_regularizer
      self.input_skip_connection = skip_connection

      size0 = self.input_size // (2 ** len(features))
      assert(features[-1] * size0 >= sum(self.latent_sizes) and
         "Final conv layer's output is smaller then latent space")

      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x

      if resnet:
         self.enc_block = self.enc_residual_block
         self.dec_block = self.dec_residual_block
      else:
         self.enc_block = self.enc_normal_block
         self.dec_block = self.dec_normal_block

      if isinstance(decoder_noise, (int, float)):
         self.dec_noise = lambda: F.noise(decoder_noise)
      elif isinstance(decoder_noise, dict):
         dropout_value = decoder_noise.get("dropout")
         if dropout_value:
            self.dec_noise = lambda: F.dropout(dropout_value)
         else:
            self.dec_noise = lambda: F.noise(**decoder_noise)
      else:
         self.dec_noise = lambda: identity


   @staticmethod
   def up1d(factor=2):
      return fun._ >> L.UpSampling1D(factor)

   def enc_residual_block(self, n_features_in, n_features_out, name=None):
      act = self.act
      def block_f(x):
         #block = F.conv1d(n_features_out, self.kernel_size) >> DownSampling1D(2) >> act() >> self.batch_norm()
         block = F.conv1d(n_features_out, self.kernel_size) >> act() >> F.conv1d(n_features_out, self.kernel_size, 2)
         blockx = block(x)
         if n_features_out != n_features_in:
            skip_connection = F.conv1d(n_features_out, 1, stride=2)
         else:
            skip_connection = DownSampling1D(2)
         # if n_features_out < n_features_in:
         #    skip_connection = K.stack
         # elif n_features_out > n_features_in:
         #    skip_connection = skip_connection[:,:,:n_features_out]
         return L.add([blockx, skip_connection(x)])
      return fun._ >> block_f >> act()

   def enc_normal_block(self, n_features_in, n_features_out, name=None):
      act = self.act

      # def block_f(x):
      #    skip_connection = DownSampling1D(2)
      #    block = (  F.conv1d(n_features_out, self.kernel_size, 2, name=name+'.conv') >> act(name+'.act')
      #            >> self.batch_norm()
      #            )
      #    return L.add([block(x), skip_connection(x)])
      # return fun._ >> block_f

      if self.one_one_conv:
         return (  F.conv1d(n_features_out, self.kernel_size, 1, name=name+'.conv1') >> act(name+'.act')
                >> F.conv1d(n_features_out, 1, 2, name=name+'.conv2') >> act()
                >> self.batch_norm()
                )
      else:
         return (  F.conv1d(n_features_out, self.kernel_size, 2, name=name+'.conv') >> act(name+'.act')
                >> self.batch_norm()
                )

   def make_encoder(self, latent_size=None, name_prefix=''):
      # assert(type(latent_size) in (None, list, tuple))
      # assert(type(name_prefix) in (None, str))

      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      block = self.enc_block
      act = self.act

      feat_in = np.concatenate([[1], self.features[:-1]])
      feat_out = self.features
      blocks = [block(i, o, name_prefix+'enc.' + str(n)) for n, (i, o) in enumerate(zip(feat_in, feat_out))]

      if self.input_skip_connection:
         # merge the input into each layer from a parallel cascade of downsamplers
         downsamplinging_cascade = [fun._ >> DownSampling1D(2) for _ in range(len(self.features))]

         def compose(f, gs):
            def c(x1, x2):
               y1, y2 = f(x1, x2)
               m = L.concatenate([y1, y2])
               return (gs[0](m), gs[1](y2))
            return c
         ladder = reduce(
            compose,
            [(b,d) for (b, d) in zip(blocks, downsamplinging_cascade)],
            lambda x1, x2: (x1, x2)
         )
         conv_blocks = lambda x: ladder(x, x)[0]
      else:
         conv_blocks = reduce(lambda x, y: x >> y, blocks)

      return (  F.append_dimension()
             >> conv_blocks
             >> F.flatten()
             >> F.dense([latent_size]) >> act()
             # >> F.dense([latent_size], activity_regularizer=keras.regularizers.l1(0.0002)) >> act()
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.01, no_sampling=True)
             )


   def dec_residual_block(self, n_features_in, n_features_out):
      act = self.act
      def block_f(x):
         up = self.up1d
         block = up(2) >> F.conv1d(n_features_out, self.kernel_size) >> act() >> F.conv1d(n_features_out, self.kernel_size)
         blockx = block(x)
         if n_features_out != n_features_in:
            skip_connection = F.conv1d(n_features_out, 1)
         else:
            skip_connection = up(2)
         return L.add([blockx, skip_connection(x)])
      #return self.up1d() >> block_f >> act()
      return fun._ >> block_f >> act() >> self.dec_noise()

   def up_conv(self, n_features, kernel_size):
      if self.upsample_with_zeros:
         return F.conv1d_transpose(n_features, kernel_size, 2, kernel_regularizer=self.decoder_kernel_regularizer)
      else:
         return self.up1d(2) >> F.conv1d(n_features, self.kernel_size)

   def dec_normal_block(self, _, n_features_out):
      act = self.act

      # def block_f(x):
      #    block = self.up_conv(n_features_out, self.kernel_size)
      #    skip_connection = self.up1d(2)
      #    return L.add([block(x), skip_connection(x)])
      # return fun._ >> block_f

      if self.one_one_conv:
          block = (  self.up_conv(n_features_out, self.kernel_size) >> act()
                  >> F.conv1d(n_features_out, 1)
                  )
      else:
         block = self.up_conv(n_features_out, self.kernel_size)
      return block >> act() >> self.batch_norm() >> self.dec_noise()


   def make_decoder(self, custom_features=None, custom_noise=None):
      # hack FIXME:
      if custom_noise: self.dec_noise = custom_noise

      act = self.act
      up = self.up1d
      features = custom_features or self.dec_features
      block = self.dec_block
      size0 = self.input_size // (2 ** len(features))

      conv_blocks = reduce(
         lambda x, y: x >> y,
         [block(i, o) for (i, o) in zip(features[:-1], features[1:])]
      )

      return (  F.dense([size0, features[0]]) >> act() >> self.batch_norm() # >> F.noise(0.5)
             >> conv_blocks
             >> up() >> F.conv1d(1, self.final_kernel_size)
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



class BlockFactory:

   def __init__(self, input_size, latent_sizes, kernel_size=5, features=[4, 4, 8, 8, 16, 16, 16],
      use_batch_norm=False, scale_factor=1
   ):
      self.input_size = input_size
      self.latent_sizes = latent_sizes
      self.kernel_size = kernel_size
      self.features = features
      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x
      self.scale_factor = scale_factor

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

   def make_decoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size

      # VERSION 1: output are features
      # return (  F.append_dimension()
      #        >> F.conv1d(2, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(4, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(8, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(16, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(32, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(64, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(128, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
      #        >> F.conv1d(128, 1,  1)
      #        >> F.flatten()
      #        )

      # VERSION 2: transform to features (time size 1) and back to time 128 (features 1)
      up = self.up1d
      return (  F.append_dimension()
             >> F.conv1d(2, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> F.conv1d(4, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> F.conv1d(8, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> F.conv1d(16, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> F.conv1d(32, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> F.conv1d(64, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> F.conv1d(128, ks, 2) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(64, ks) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(32, ks) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(16, ks) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(8, ks) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(4, ks) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(2, ks) >> act() >> F.noise(0.2, decay=0.001, final_stddev=0.05)
             >> up() >> F.conv1d(1, ks)
             >> F.flatten()
             )

   def make_encoder(self):
      return identity

# --------------------------------------------------------------------------------------------------
# FACTOR MODEL -- actual source separator that factors the state/latent space
# --------------------------------------------------------------------------------------------------


def rotate(xs):
   yield xs[-1]
   for x in xs[:-1]:
      yield x


def make_factor_model(
   example_frame,
   factory,
   input_noise=None,
   latent_noise=None,
   shared_encoder=True,
   custom_features=None,
   custom_noises=None,
   vanishing_xprojection=None,
   info_loss=None
):

   x = F.input_like(example_frame)
   #x_2 = F.input_like(example_frame)    # The Variational layer causes conflicts if this is in and not connected
   #eta2 = lambda: F.noise(0.1)

   if isinstance(input_noise, (int, float)):
      eta = lambda: F.noise(input_noise)
   elif isinstance(input_noise, dict):
      eta = lambda: F.noise(**input_noise)
   else:
      eta = lambda: fun._

   if isinstance(latent_noise, (int, float)):
      latent_eta = lambda: F.noise(latent_noise)
   elif isinstance(latent_noise, dict):
      latent_eta = lambda: F.noise(**latent_noise)
   else:
      latent_eta = lambda: fun._


   if shared_encoder:
      encoder = factory.make_encoder()
      cum_latent_sizes = np.concatenate([[0], np.cumsum(factory.latent_sizes)])

      def make_slice(n,m):
         def s(x):
            slice = XL.Slice[:, n:m]
            sx = slice(x)
            if info_loss:
               compressor = (  F.dense([m-n]) >> L.Activation('tanh')
                            >> F.dense([1]) >> L.Activation('tanh')
                            >> F.dense([m-n]) >> L.Activation('tanh')
                            >> F.dense([m-n]) )
               #compressor = F.dense([1]) >> L.Activation('tanh') >> F.dense([m-n])
               slice.add_loss(info_loss * K.mean(K.square(sx - compressor(sx))))
            return sx
         return fun._ >> s

      if not custom_features:
         custom_features = [None for _ in factory.latent_sizes]
      if not custom_noises:
         custom_noises = [None for _ in factory.latent_sizes]
      assert( len(custom_features) == len(factory.latent_sizes) )
      assert( len(custom_noises) == len(factory.latent_sizes) )

      decoders = [
         make_slice(cum_latent_sizes[n], cum_latent_sizes[n+1]) >> factory.make_decoder(f, e)
         # for DILATED: fun._ >> XL.Slice[:,:, cum_latent_sizes[n]:cum_latent_sizes[n+1]] >> factory.make_decoder()
         #for n in range(len(factory.latent_sizes))
         for n,(f,e) in enumerate(zip(custom_features, custom_noises))
      ]
      ex = (eta() >> encoder)(x)
      #ex.add_loss(keras.losses.mean_absolute_error())
      channels = [(latent_eta() >> dec)(ex) for dec in decoders]
   else:
      encoders = [factory.make_encoder(z) for z in factory.latent_sizes]
      decoders = [factory.make_decoder() for n in range(len(factory.latent_sizes))]
      channels = [(eta() >> enc >> latent_eta() >> dec)(x) for enc, dec in zip(encoders, decoders)]
      ex = L.concatenate([e(x) for e in encoders])

   #channels = [(dec >> eta2() >> encoder >> dec)(ex) for dec in decoders]

   residual_separator = False
   if residual_separator:
      cum_latent_sizes = np.concatenate([[0], np.cumsum(factory.latent_sizes)])
      enc = factory.make_encoder()
      slice1 = XL.Slice[:, cum_latent_sizes[0]:cum_latent_sizes[1]]
      slice2 = XL.Slice[:, cum_latent_sizes[1]:cum_latent_sizes[2]]
      dec1 = factory.make_decoder()
      dec2 = factory.make_decoder()

      encB = factory.make_encoder(name_prefix='res')
      slice1B = XL.Slice[:, cum_latent_sizes[0]:cum_latent_sizes[1]]
      slice2B = XL.Slice[:, cum_latent_sizes[1]:cum_latent_sizes[2]]
      dec1B = factory.make_decoder()
      dec2B = factory.make_decoder()

      exA = (eta() >> enc)(x)
      nexA = exA
      z1A = slice1(nexA)
      z2A = slice2(nexA)
      channelsA = [dec1(slice1(nexA)), dec2(slice2(nexA))]
      u = L.add(channelsA)

      latent_skip = False
      exB = (eta() >> encB)(L.subtract([x, u]))
      if latent_skip: exB = L.add([exB, exA])
      channelsB = [dec1B(slice1B(exB)), dec2B(slice2B(exB))]

      xu = L.add(channelsB)
      y = L.add([u, xu])

      ex = exA  # TODO use exB and add before slices
      channels = [L.add([channelsB[0], channelsA[0]]), L.add([channelsB[1], channelsA[1]])]

   y = L.add(channels)

   m = M.Model([x], [y])
   #m.add_loss(10*K.mean( K.square(ex[:,0:2])) * K.mean(K.square(ex[:,2:])))
   # for z1 in range(0, factory.latent_sizes[0]):
   #    for z2 in range(factory.latent_sizes[0], factory.latent_sizes[0] + factory.latent_sizes[1]):
   #       m.add_loss(0.2 * K.square(K.mean(ex[:, z1] * ex[:, z2])))


   #ex_2 = (eta() >> encoder)(x_2)
   #m_slow_feat = M.Model([x, x_2], [y])
   #m_slow_feat.add_loss(1*K.mean(K.square( ex_2[:,0:2] - ex[:,0:2] + ex_2[:,4:6] - ex[:,4:6] )))

   # for c in channels:
   #    compressor = F.dense([3]) >> act() >> F.dense([example_frame.shape[0]])
   #    m.add_loss(.3*K.mean(K.square(c - compressor(c))))

   if vanishing_xprojection == True:
      vanishing_xprojection = 0.2
   if vanishing_xprojection:
      # for c1, c2 in zip(channels, rotate(channels)):
      #    m.add_loss(1*K.mean(K.square(c1(c2))))
      for d,c in zip(decoders, rotate(channels)):
         m.add_loss(vanishing_xprojection * K.mean(K.square((encoder >> d)(c))))
      # for d,c in zip(decoders, rotate(channels)):
      #    static_channel = M.Model([x], [(encoder >> d)])
      #    m.add_loss(0.2*K.mean(K.square((static_channel)))

   # another idea: minimizing the energy per channel should support separation
   #  for c in channels:
   #    m.add_loss(0.1 * K.mean(K.square(c)))


   return (
      m,
      M.Model([x], [ex]),
      #M.Model([x], [(encoder >> F.flatten())(x)]),
      None,#m_slow_feat,
      [M.Model([x], [c]) for c in channels]
   )
