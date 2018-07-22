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

# TERMS
# • factorization

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


factor = 1
frame_size = factor*128
shift = 8
n_pairs = 10000
n_latent1 = 4
n_latent2 = 4
n_epochs = 20
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

   # IDEAS:
   # • constraint on separated channels ? e.g. less fluctuations
   # • SAE with with separation in z-space

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




def make_conv_model(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   x_2 = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   kernel_size = 3
   #features = [4, 8, 8, 16, 32, 32, 32]
   features = [4, 4, 8, 8, 16, 16, 16]
   #features = [2, 4, 4, 4, 8, 8, 8]

   latent_size = sum(latent_sizes)
   latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])

   def up1d(factor=2):
      #return fun._ >> UpSampling1DZeros(factor)
      return fun._ >> L.UpSampling1D(factor)

   # try skip layer / residuals

   def make_encoder():
      return (  F.reshape([sig_len,1])
             >> F.conv1d(features[0], kernel_size, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[1], kernel_size, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[2], kernel_size, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[3], kernel_size, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[4], kernel_size, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[5], kernel_size, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(features[6], kernel_size, 2) >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> dense([latent_size]) >> act()
             )

   def make_decoder(n):
      return (  fun._ >> XL.Slice[:, latent_sizes2[n]:latent_sizes2[n+1]]
              >> dense([factor, features[6]]) >> act()
              >> up1d() >> F.conv1d(features[5], kernel_size) >> act() #>> F.batch_norm()
              >> up1d() >> F.conv1d(features[4], kernel_size) >> act() #>> F.batch_norm()
              >> up1d() >> F.conv1d(features[3], kernel_size) >> act() #>> F.batch_norm()
              >> up1d() >> F.conv1d(features[2], kernel_size) >> act() #>> F.batch_norm()
              >> up1d() >> F.conv1d(features[1], kernel_size) >> act() #>> F.batch_norm()
              >> up1d() >> F.conv1d(features[0], kernel_size) >> act() #>> F.batch_norm()
              >> up1d() >> F.conv1d(1, kernel_size)
              >> F.flatten()
              )

   encoder = make_encoder()
   decoder1 = make_decoder(0)
   decoder2 = make_decoder(1)


   ex = (eta() >> encoder)(x)
   decoders = [make_decoder(n) for n in range(len(latent_sizes))]
   channels = [dec(ex) for dec in decoders]
   #channels = [(dec >> eta() >> encoder >> dec)(ex) for dec in decoders]
   y = L.add(channels)

   m = M.Model([x], [y])
   #m.add_loss(10*K.mean( K.square(ex[:,0:2])) * K.mean(K.square(ex[:,2:])))


   ex_2 = (eta() >> encoder)(x_2)
   m_slow_feat = M.Model([x, x_2], [y])
   m_slow_feat.add_loss(1*K.mean(K.square( ex_2[:,0:2] - ex[:,0:2] + ex_2[:,4:6] - ex[:,4:6] )))

   # m.add_loss(1*K.mean(K.square((encoder >> slice1 >> decoder1)(y2))))
   # m.add_loss(1*K.mean(K.square((encoder >> slice2 >> decoder2)(y1))))

   return (
      m,
      M.Model([x], [encoder(x)]),
      m_slow_feat,
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



#make_2freq = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(np.pi*0.05*np.arange(n))
sin0 = lambda n: 0.3*np.sin(0.03*np.arange(n))
sin1 = lambda n: 0.64*np.sin(0.05*np.arange(n))
sin2 = lambda n: 0.3*np.sin(np.pi*0.05*np.arange(n))
sin1exp = lambda n: sin1(n) * np.exp(-0.001*np.arange(n))
sin2am = lambda n: sin2(n) * (1+0.4*np.sin(0.021231*np.arange(n)))
kick1 = lambda n: np.sin( 100*np.exp(-0.001*np.arange(n)) ) * np.exp(-0.001*np.arange(n))
kick2 = lambda n: np.sin( 250*np.exp(-0.002*np.arange(n)) ) * np.exp(-0.001*np.arange(n))
tanhsin1 = lambda n: 0.6*np.tanh(4*np.sin(0.05*np.arange(n)))
fm_soft = lambda n: np.sin(0.07*np.arange(n) + 4*np.sin(0.00599291*np.arange(n)))
fm_soft1 = lambda n: np.sin(np.pi*0.05*np.arange(n) + 3*np.sin(0.00599291*np.arange(n)))
fm_soft3 = lambda n: np.sin(np.pi*0.1*np.arange(n) + 6*np.sin(0.00599291*np.arange(n)))
fm_soft3inv = lambda n: np.sin(np.pi*0.1*np.arange(n) - 6*np.sin(0.00599291*np.arange(n)))
fm_soft2 = lambda n: np.sin(0.15*np.arange(n) + 18*np.sin(0.00599291*np.arange(n)))
fm_med = lambda n: np.sin(0.1*np.arange(n) + 1*np.sin(0.11*np.arange(n)))
fm_strong = lambda n: 0.5*np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
fm_strong2 = lambda n: 0.5*np.sin(0.06*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
fm_hyper = lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)) + 2*np.sin(0.009*np.arange(n)))
lorenz = lambda n: TS.lorenz(n, [1,0,0])[::1]
lorenz2 = lambda n: TS.lorenz(n+25000, [0,-1,0])[25000:]

#sig1 = kick2
#sig2 = sin2
sig1 = lambda n: 0.3*lorenz(n)
sig2 = lambda n: 0.3*fm_strong(n)
#sig2 = sin0
#sig2 = lambda n: 0.3*fm_soft1(n)
#sig2 = lambda n: 0.3*lorenz2(n)

#sig1 = lambda n: 0.3*fm_soft3(n)
#sig2 = lambda n: 0.3*fm_soft3inv(n)
make_2freq = lambda n: sig1(n) + sig2(n)

frames, out_frames, *_ = TS.make_training_set(make_2freq, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

frames1, *_ = TS.make_training_set(sig1, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
frames2, *_ = TS.make_training_set(sig2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)




#trainer, model, model2, mode1, mode2, encoder, model_sf = make_model(frames[0])
model, encoder, model_sf, [mode1, mode2] = make_conv_model(frames[0])
#_, model, model2, mode1, mode2, encoder, encoder2 = make_model2(frames[0])
loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) #+ 0.001*K.sum(dzdx*dzdx)

model.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
model_sf.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
model.summary()
plot_model(model, to_file='ssae.png', show_shapes=True)

x = F.input_like(frames[0])
cheater = M.Model([x], [mode1(x), mode2(x)])
cheater.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)

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
# def plot_modes3(n=2000):
#    figure()
#    plot(sig1(n), 'k')
#    plot(sig2(n), 'k')
#    plot(build_prediction(mode1, frames, n), 'r')
#    plot(build_prediction(mode2, frames, n), 'r')
# 
# def plot_modes2(n=2000):
#    figure()
#    plot(build_prediction(mode1, n))
#    plot(build_prediction(mode2, n))
# 
# def plot_orig_vs_reconst(n=2000):
#    plot(build_prediction(model, n))
#    plot(make_2freq(n))
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

training_summary(model, mode1, mode2, encoder, make_2freq, sig1, sig2, frames, loss_recorder)

# plot_modes2()
# plot_joint_dist()

# plot(P.predict_signal(model, frames[0], shift, 5000), 'k')
# plot(make_2freq(1000))
