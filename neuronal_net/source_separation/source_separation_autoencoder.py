# TODO
# • add slow feature regularization on z-subspaces
# • debug decorrelation approach


# TODO
# • separating two lorenz with current model is hard even using cheater
#   and unstable when training model afterwards
#   → try out more complex model

# TRY
# • minimize cross-channel predictivity, i.e. it should be impossible for the model to predict
#   the x2 out z1 and vise versa
#   • perhaps formulate in terms of ShAE, i.e. it's not possible to build a sub-ShAE across
#     the channels
#   • adversarial style training.
# • simplify (remove layers) from decoder
# • simplify parallel AE
# • multi-pass ShAE

# IDEAS:
# • constraint on separated channels ? e.g. less fluctuations
# • SAE with with separation in z-space


import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.utils import plot_model

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS
from keras_tools.upsampling import UpSampling1DZeros

from timeshift_autoencoder import predictors as P
from result_tools import *
from test_data import *
from entropy import naive_mutual_information, mutual_information


factor = 1
frame_size = factor*128
shift = 8
n_pairs = 20000
n_latent1 = 3
n_latent2 = 3
latent_sizes = [n_latent1, n_latent2]
n_epochs = 10
noise_stddev = 0.05


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

   def make_decoder(self, n):
      return (  fun._ >> XL.Slice[:, self.latent_sizes2[n]:self.latent_sizes2[n+1]]
              >> F.dense([self.input_size//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([self.input_size//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([self.input_size])
              )



class ConvFactory:

   def __init__(self, example_frame, latent_sizes, use_batch_norm=False):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.kernel_size = 5
      #self.features = [4, 8, 8, 16, 32, 32, 32]
      self.features = [4, 4, 8, 8, 16, 16, 16]
      #self.features = [2, 4, 4, 4, 8, 8, 8]
      if use_batch_norm:
         self.batch_norm = F.batch_norm
      else:
         self.batch_norm = lambda: lambda x: x

      # try skip layer / residuals

   @staticmethod
   def up1d(factor=2):
      #return fun._ >> UpSampling1DZeros(factor)
      return fun._ >> L.UpSampling1D(factor)

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      return (  F.append_dimension()
             >> F.conv1d(features[0], ks, 2)  >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], ks, 2)  >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], ks, 2)  >> act() >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[5], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[6], ks, 2) >> act()  >> self.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> F.dense([latent_size]) >> act()
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.01, no_sampling=True)
             )

   def make_decoder(self):
      up = self.up1d
      features = self.features
      ks = self.kernel_size
      return (  F.dense([factor, features[6]]) >> act()
             >> up() >> F.conv1d(features[5], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[4], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[3], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[2], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[1], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(features[0], ks) >> act() >> self.batch_norm()
             >> up() >> F.conv1d(1, ks)
             >> F.flatten()
             )


def rotate(xs):
   yield xs[-1]
   for x in xs[:-1]:
      yield x


def make_factor_model(example_frame, factory, shared_encoder=True):

   x = F.input_like(example_frame)
   #x_2 = F.input_like(example_frame)    # The Variational layer causes conflicts if this is in and not connected
   eta = lambda: F.noise(noise_stddev)
   eta2 = lambda: F.noise(0.1)


   if shared_encoder:
      encoder = factory.make_encoder()
      cum_latent_sizes = np.concatenate([[0], np.cumsum(latent_sizes)])
      decoders = [
         fun._ >> XL.Slice[:, cum_latent_sizes[n]:cum_latent_sizes[n+1]] >> factory.make_decoder()
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

   # for d,c in zip(decoders, rotate(channels)):
   #    m.add_loss(1*K.mean(K.square((encoder >> d)(c))))

   return (
      m,
      M.Model([x], [ex]),
      None,#m_slow_feat,
      [M.Model([x], [c]) for c in channels]
   )



class LossRecorder(keras.callbacks.Callback):

   def __init__(self, **kargs):
      super(LossRecorder, self).__init__(**kargs)
      self.losses = []
      self.grads = []
      self.pred_errors = []
      #self.mutual_information = []

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights

   def on_epoch_end(self, epoch, logs={}):
      self.pred_errors.append(
      [   pred_error(mode1, frames, sig1, 2048)
      ,   pred_error(mode1, frames, sig2, 2048)
      ,   pred_error(mode2, frames, sig2, 2048)
      ,   pred_error(mode2, frames, sig1, 2048)
      ])
      #self.mutual_information.append(
      #   mutual_information(
      #      build_prediction(mode1, frames, 2000),
      #      build_prediction(mode2, frames, 2000)
      #   )
      #)


#sig1, sig2 = two_sin
#sig1, sig2 = kicks_sin1
sig1, sig2 = lorenz_fm
#sig1, sig2 = fm_twins
#sig1, sig2 = tanhsin1, sin2
#sig1, sig2 = tanhsin1, sin4
#sig1, sig2 = cello, clarinet
#sig1, sig2 = cello_dis3, choir_e4

sig_gen = sig1 + sig2
sig_gen_s = lambda n: sig1(n) + sig2(n+100)[100:]

#TRY: project extracted mode for extracted_mode + noise

frames, *_ = TS.make_training_set(sig_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

frames1, *_ = TS.make_training_set(sig1, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
frames2, *_ = TS.make_training_set(sig2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)




#trainer, model, model2, mode1, mode2, encoder, model_sf = make_model(frames[0])
#factory = DenseFactory
factory = ConvFactory
model, encoder, model_sf, [mode1, mode2] = make_factor_model(
   frames[0], factory(frames[0], latent_sizes, use_batch_norm=False), shared_encoder=True)
#_, model, model2, mode1, mode2, encoder, encoder2 = make_model2(frames[0])
loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) #+ 0.001*K.sum(dzdx*dzdx)

model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001), loss=loss_function)
#model_sf.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
model.summary()
plot_model(model, to_file='ssae.png', show_shapes=True)

#x = F.input_like(frames[0])
#cheater = M.Model([x], [mode1(x), mode2(x)])
#cheater.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)

#model2.compile(optimizer=keras.optimizers.Adam(), loss='mse')
#trainer.compile(optimizer=keras.optimizers.Adam(0.0001,0.5), loss=lambda y_true, y_pred:y_pred)
#mode1.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)



loss_recorder = LossRecorder()

tools.train(model, frames, frames, 128, 1*n_epochs, loss_recorder)
#tools.train(model_sf, [frames[:-1], frames[1:]], frames[:-1], 128, 1*n_epochs, loss_recorder)
#tools.train(cheater, frames, [frames1, frames2], 32, n_epochs, loss_recorder)


from pylab import *

def plot_modes3(n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(mode1, frames, n), 'r')
   plot(build_prediction(mode2, frames, n), 'r')

code = encoder.predict(frames)

training_summary(model, mode1, mode2, encoder, sig_gen, sig1, sig2, frames, loss_recorder)



from sklearn.decomposition import FastICA, PCA

def ica(x, n_components, max_iter=1000):
   ica_trafo = FastICA(n_components=n_components, max_iter=max_iter)
   return ica_trafo.fit_transform(x)
