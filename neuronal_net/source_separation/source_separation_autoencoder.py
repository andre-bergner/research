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


factor = 1
frame_size = factor*128
shift = 8
n_pairs = 10000
n_latent1 = 4
n_latent2 = 4
latent_sizes = [n_latent1, n_latent2]
n_epochs = 10
noise_stddev = 0.05


activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)
#act = lambda: L.LeakyReLU(alpha=0.2)

dense = F.dense
#dense = fun._ >> fun.bind(
#def dense(out_shape, *args, **kwargs):
#   return fun._ >> L.Dense(units=out_shape[0], kernel_initializer='he_normal')#, bias_initializer=keras.initializers.he_normal)
#dense = fun.bind(
#   F.dense,
#   kernel_initializer=keras.initializers.he_normal
#   bias_initializer=keras.initializers.he_normal
#)


def make_model(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)


   encoder = (  dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
             >> dense([sig_len//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             #>> XL.VariationalEncoder(latent_size, sig_len, beta=0.1)
             )

   slice1 = XL.Slice[:,0:latent_sizes[0]]
   #slice1.activity_regularizer = lambda x: 1. / (10. + K.mean(K.square((x))))
   #slice1.activity_regularizer = lambda x: K.exp(-K.mean(K.square((x))))
   decoder1 = (  dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len]) 
              )

   slice2 = XL.Slice[:,latent_sizes[0]:]
   #slice2.activity_regularizer = lambda x: 1. / (0.001 + K.mean(K.square((x))))
   decoder2 = (  dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len]) 
              )


   # y = eta() >> encoder >> eta() >> decoder
   #y1 = eta() >> encoder >> eta() >> decoder1
   #y2 = eta() >> encoder >> eta() >> decoder2
   ex = encoder(x)
   z1 = slice1(ex)
   z2 = slice2(ex)
   y1 = decoder1(z1)
   y2 = decoder2(z2)
   #y1 = (eta() >> encoder >> slice1 >> decoder1 >> eta() >> encoder >> slice1 >> decoder1)(x)
   #y2 = (eta() >> encoder >> slice2 >> decoder2 >> eta() >> encoder >> slice2 >> decoder2)(x)
   #y = lambda x: L.Add()([y1(x), y2(x)])
   y = L.Add()([y1, y2])

   #dzdx = XL.jacobian(ex,x)

   #distangle_pressure = lambda -keras.losses.mean_squared_error()
   #reconstruction_loss = keras.losses.mean_squared_error
   def loss_f(args):
      y_true, y_pred = args
      l2 = K.mean(K.square(y_true - y_pred))
      #return l2
      #return l2 + 0.002*K.mean(y1(x) * y2(x))
      #return l2 + 0.1 * K.exp(-K.square(K.mean(y1(x) - y2(x))))
      #return l2 + 10*K.square(K.mean(y1(x) * y2(x))) / ( K.mean(K.square(y1(x))) * K.mean(K.square(y2(x))) )
      #return l2 + .1 * K.square(K.mean( (y1(x)-K.mean(y1(x))) * (y2(x)-K.mean(y2(x))) )) \
      #               / ( K.mean(K.square(y1(x)-K.mean(y1(x)))) * K.mean(K.square(y2(x)-K.mean(y2(x)))) )
      #Y1 = z1(x) - K.mean(z1(x))
      #Y2 = z2(x) - K.mean(z2(x))
      Y1 = y1 - K.mean(y1)
      Y2 = y2 - K.mean(y2)
      return l2 + 0.1 * K.square( K.mean(Y1 * Y2) / (K.mean(K.square(Y1)) * K.mean(K.square(Y2))) )
      #return l2 / (1. + K.tanh(K.mean(K.square(y1(x) - y2(x)))))
      #return l2 - 0.01 * K.tanh(0.01*K.mean(K.square(y1(x) - y2(x))))
      #return l2 / (100. + K.mean(K.square(y1(x) - y2(x))))
      #return l2 / (10. + K.mean(K.square(y1(x) / (1.+K.abs(y2(x))) + y2(x) / (1.+K.abs(y1(x)))     )))

   loss = L.Lambda(loss_f, output_shape=(1,))

   m = M.Model([x], [y])
   m.add_loss(10*K.mean(K.square((encoder >> slice1 >> decoder1)(y2))))
   m.add_loss(10*K.mean(K.square((encoder >> slice2 >> decoder2)(y1))))

   return (
      M.Model([x], [loss([x, y])]),
      m,#M.Model([x], [y]),
      None,#M.Model([x], [y, y(y)]),
      M.Model([x], [y1]),
      M.Model([x], [y2]),
      M.Model([x], [encoder(x)]),
      None#dzdx
   )


class DenseFactory:

   def __init__(self, example_frame, latent_sizes):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])

   def make_encoder(self):
      latent_size = sum(self.latent_sizes)

      return (  dense([self.input_size//2])  >> act()                   # >> F.dropout(0.2)
             >> dense([self.input_size//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.1)
             )

   def make_decoder(self, n):
      return (  fun._ >> XL.Slice[:, self.latent_sizes2[n]:self.latent_sizes2[n+1]]
              >> dense([self.input_size//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([self.input_size//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([self.input_size])
              )




class ConvFactory:

   def __init__(self, example_frame, latent_sizes):
      self.input_size = example_frame.shape[-1]
      self.latent_sizes = latent_sizes
      self.latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])
      self.kernel_size = 3
      #self.features = [4, 8, 8, 16, 32, 32, 32]
      self.features = [4, 4, 8, 8, 16, 16, 16]
      #self.features = [2, 4, 4, 4, 8, 8, 8]

      # try skip layer / residuals

   @staticmethod
   def up1d(factor=2):
      #return fun._ >> UpSampling1DZeros(factor)
      return fun._ >> L.UpSampling1D(factor)

   def make_encoder(self):
      latent_size = sum(self.latent_sizes)
      features = self.features
      ks = self.kernel_size
      return (  F.append_dimension()
             >> F.conv1d(features[0], ks, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], ks, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], ks, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[5], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[6], ks, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> dense([latent_size]) >> act()
             #>> XL.VariationalEncoder(latent_size, self.input_size, beta=0.01)
             )

   def make_decoder(self, n):
      up = self.up1d
      features = self.features
      ks = self.kernel_size
      return (  fun._ >> XL.Slice[:, self.latent_sizes2[n]:self.latent_sizes2[n+1]]
              >> dense([factor, features[6]]) >> act()
              >> up() >> F.conv1d(features[5], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[4], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[3], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[2], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[1], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(features[0], ks) >> act() #>> F.batch_norm()
              >> up() >> F.conv1d(1, ks)
              >> F.flatten()
              )



def make_factor_model(example_frame, factory):

   x = F.input_like(example_frame)
   #x_2 = F.input_like(example_frame)    # The Variational layer causes conflicts if this is in and not connected
   eta = lambda: F.noise(noise_stddev)

   encoder = factory.make_encoder()
   ex = (eta() >> encoder)(x)
   decoders = [factory.make_decoder(n) for n in range(len(factory.latent_sizes))]
   channels = [dec(ex) for dec in decoders]
   #channels = [(dec >> eta() >> encoder >> dec)(ex) for dec in decoders]
   y = L.add(channels)

   m = M.Model([x], [y])
   #m.add_loss(10*K.mean( K.square(ex[:,0:2])) * K.mean(K.square(ex[:,2:])))

   #ex_2 = (eta() >> encoder)(x_2)
   #m_slow_feat = M.Model([x, x_2], [y])
   #m_slow_feat.add_loss(1*K.mean(K.square( ex_2[:,0:2] - ex[:,0:2] + ex_2[:,4:6] - ex[:,4:6] )))

   # m.add_loss(1*K.mean(K.square((encoder >> slice1 >> decoder1)(y2))))
   # m.add_loss(1*K.mean(K.square((encoder >> slice2 >> decoder2)(y1))))

   return (
      m,
      M.Model([x], [encoder(x)]),
      None,#m_slow_feat,
      [M.Model([x], [c]) for c in channels]
   )





def make_model2(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)


   encoder1 = (  F. dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
              >> F. dense([sig_len//4])  >> act() >> F.batch_norm() # >> F.dropout(0.2)
              #>> F. dense([latent_sizes[0]]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> XL.VariationalEncoder(latent_sizes[0], sig_len, beta=0.002)
              )

   decoder1 = (  F.dense([sig_len//4]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   encoder2 = (  F. dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
              >> F. dense([sig_len//4])  >> act() >> F.batch_norm() # >> F.dropout(0.2)
              #>> F. dense([latent_sizes[1]]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> XL.VariationalEncoder(latent_sizes[1], sig_len, beta=0.002)
              )

   decoder2 = (  F.dense([sig_len//4]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   # IDEAS:
   # • constraint on separated channels ? e.g. less fluctuations
   # • SAE with with separation in z-space

   # y = eta() >> encoder >> eta() >> decoder
   y1 = eta() >> encoder1 >> eta() >> decoder1
   y2 = eta() >> encoder2 >> eta() >> decoder2
   y = lambda x: L.Add()([y1(x), y2(x)])

   return (
      None,#M.Model([x], [loss([x, y(x)])]),
      M.Model([x], [y(x)]),
      M.Model([x], [y(x), y(y(x))]),
      M.Model([x], [y1(x)]),
      M.Model([x], [y2(x)]),
      M.Model([x], [encoder1(x)]),
      M.Model([x], [encoder2(x)]),
   )

class LossRecorder(keras.callbacks.Callback):

   def __init__(self, **kargs):
      super(LossRecorder, self).__init__(**kargs)
      self.losses = []
      self.grads = []
      self.pred_errors = []

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


#sig1, sig2 = kicks_sin1
sig1, sig2 = lorenz_fm

#sig2 = sin0
#sig2 = 0.3*fm_soft1(n)
#sig2 = 0.3*lorenz2(n)

#sig1 = lambda n: 0.3*fm_soft3(n)
#sig2 = lambda n: 0.3*fm_soft3inv(n)
sig_gen = sig1 + sig2

frames, out_frames, *_ = TS.make_training_set(sig_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

frames1, *_ = TS.make_training_set(sig1, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
frames2, *_ = TS.make_training_set(sig2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)




#trainer, model, model2, mode1, mode2, encoder, model_sf = make_model(frames[0])
#factory = DenseFactory
factory = ConvFactory
model, encoder, model_sf, [mode1, mode2] = make_factor_model(frames[0], factory(frames[0], latent_sizes))
#_, model, model2, mode1, mode2, encoder, encoder2 = make_model2(frames[0])
loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) #+ 0.001*K.sum(dzdx*dzdx)

model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.7, beta_2=0.9, decay=0.0001), loss=loss_function)
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
#tools.train(model, frames, out_frames[0], 32, n_epochs, loss_recorder)
#tools.train(model, frames, out_frames[0], 128, 15*n_epochs, loss_recorder)
#tools.train(cheater, frames, [frames1, frames2], 32, n_epochs, loss_recorder)
#tools.train(model2, frames, out_frames, 32, n_epochs, loss_recorder)
#tools.train(trainer, frames, frames, 32, n_epochs, loss_recorder)



from pylab import *
# 
# figure()
# semilogy(loss_recorder.losses)
# 
# 
# def plot_modes(n=0):
#    figure()
#    plot(frames[n], 'k')
#    plot(model.predict(frames[n:n+1])[0], 'b')
#    plot(mode1.predict(frames[n:n+1])[0], 'r')
#    plot(mode2.predict(frames[n:n+1])[0], 'r')
#    plot(sig1(frame_size+n)[n:n+frame_size], 'g')
#    plot(sig2(frame_size+n)[n:n+frame_size], 'g')
# 
# 
def plot_modes3(n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(mode1, frames, n), 'r')
   plot(build_prediction(mode2, frames, n), 'r')
# 
# def plot_modes2(n=2000):
#    figure()
#    plot(build_prediction(mode1, n))
#    plot(build_prediction(mode2, n))
# 
# def plot_orig_vs_reconst(n=2000):
#    plot(build_prediction(model, n))
#    plot(sig_gen(n))
# 
# def plot_joint_dist():
#    code = encoder.predict(frames)
#    fig, axs = plt.subplots(n_latent1+n_latent2, n_latent1+n_latent2, figsize=(8, 8))
#    for ax_rows, c1 in zip(axs, code.T):
#       for ax, c2 in zip(ax_rows, code.T):
#          ax.plot( c2, c1, '.k', markersize=0.5)
#          ax.axis('off')
# 
# def plot_joint_dist2():
#    code1 = encoder.predict(frames)
#    code2 = encoder2.predict(frames)
#    code = np.concatenate([code1.T,code2.T]).T
#    fig, axs = plt.subplots(n_latent1+n_latent2, n_latent1+n_latent2, figsize=(8, 8))
#    for ax_rows, c1 in zip(axs, code.T):
#       for ax, c2 in zip(ax_rows, code.T):
#          ax.plot( c2, c1, '.k', markersize=0.5)
#          ax.axis('off')


code = encoder.predict(frames)

training_summary(model, mode1, mode2, encoder, sig_gen, sig1, sig2, frames, loss_recorder)

# plot_modes2()
# plot_joint_dist()

# plot(P.predict_signal(model, frames[0], shift, 5000), 'k')
# plot(sig_gen(1000))
