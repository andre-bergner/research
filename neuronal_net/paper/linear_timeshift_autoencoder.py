import numpy as np
import pylab as pl

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


frame_size = 128
shift = 8
noise_stddev = 0.05
n_pairs = 5000
n_epochs = 200


# make_1freq = lambda n: np.sin(0.05*np.arange(n))
# n_latent = 2

#make_2freq = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(0.1623*np.arange(n))
make_2freq = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(np.pi*0.05*np.arange(n))
#make_2freq = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(3*0.05*np.arange(n))
n_latent = 3   # will not be able to learn both frequencies without non-linearities to fold torus into 3D
# n_latent = 4
make_1freq = make_2freq


in_frames, out_frames, *_ = TS.make_training_set(make_1freq, frame_size=frame_size, n_pairs=n_pairs, shift=shift)

def make_linear_model(example_frame, latent_size):
   frame_size = np.size(example_frame)
   x = F.input_like(example_frame)

   eta = F.noise(noise_stddev)
   encoder = F.dense([latent_size])
   decoder = F.dense([frame_size])

   y = eta >> encoder >> eta >> decoder
   latent = encoder(x)
   return M.Model([x], [y(x)]), M.Model([x], [latent]), XL.jacobian(latent,x)


def make_nonlinear_model(example_frame, latent_size):
   act = lambda: L.Activation(fun.bind(XL.tanhx, alpha=0.1))
   frame_size = np.size(example_frame)
   x = F.input_like(example_frame)

   eta = F.noise(noise_stddev)

   encoder = (  F.dense([frame_size // 2]) >> act()
             #>> F.dense([frame_size // 4]) >> act() #>> F.batch_norm()
             >> F.dense([latent_size]) >> act()     #>> F.batch_norm()
             )

   decoder = (  F.dense([frame_size // 4]) >> act() #>> F.batch_norm()
             #>> F.dense([frame_size // 2]) >> act() #>> F.batch_norm()
             >> F.dense([frame_size])
             )

   y = eta >> encoder >> eta >> decoder
   latent = encoder(x)
   return M.Model([x], [y(x)]), M.Model([x], [latent]), XL.jacobian(latent,x)


lin_model, lin_encoder, lin_dzdx = make_linear_model(in_frames[0], n_latent)
lin_loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) + 0.0001*K.sum(lin_dzdx*lin_dzdx)
lin_loss_recorder = tools.LossRecorder()
lin_model.compile(optimizer=keras.optimizers.Adam(), loss=lin_loss_function)
#tools.train(lin_model, in_frames, out_frames[0], 32, n_epochs, lin_loss_recorder)
lin_model.load_weights('lin_n128_s8_zdim3.hdf5')
#tools.train(lin_model, in_frames, out_frames[0], 128, 5*n_epochs, lin_loss_recorder)

nl_model, nl_encoder, nl_dzdx = make_nonlinear_model(in_frames[0], n_latent)
nl_loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) + 0.0001*K.sum(nl_dzdx*nl_dzdx)
nl_loss_recorder = tools.LossRecorder()
nl_model.compile(optimizer=keras.optimizers.Adam(), loss=nl_loss_function)
#tools.train(nl_model, in_frames, out_frames[0], 32, n_epochs, nl_loss_recorder)
nl_model.load_weights('nl_n128_s8_zdim3__long_train.ver2.hdf5')
#tools.train(nl_model, in_frames, out_frames[0], 128, 10*n_epochs, nl_loss_recorder)
#tools.train(nl_model, in_frames, out_frames[0], 512, 15*n_epochs, nl_loss_recorder)



def plot_prediction(model, n, signal_gen):
   sig = signal_gen(n+100)
   pred_sig = P.predict_signal2(model, sig[:frame_size], shift, n+100)
   fig, ax = pl.subplots(2, 1, figsize=(10,5))
   ax[0].plot(sig[:n], 'k', label='groud truth')
   ax[0].plot(pred_sig[:n], color='dodgerblue', linewidth=2, label='prediction')
   ax[0].set_ylabel(r'$x_n$, $\hat{x}_n$', fontsize=16)
   ax[0].set_xlim([-10,n])
   ax[0].legend(loc='upper right')
   ax[1].plot(sig[:n]-pred_sig[:n], 'k')
   ax[1].set_xlabel(r'$n$ (samples)', fontsize=16)
   ax[1].set_ylabel(r'$x_n-\hat{x}_n$', fontsize=16)
   ax[1].set_xlim([-10,n])
   pl.tight_layout()


def plot_prediction_wo_diff(model, n, signal_gen):
   sig = signal_gen(n+100)
   pred_sig = P.predict_signal2(model, sig[:frame_size], shift, n+100)
   fig = pl.figure(figsize=(8,2))
   ax = pl.gca()
   ax.plot(sig[:n], 'k', label='groud truth')
   ax.plot(pred_sig[:n], color='dodgerblue', linewidth=2, label='prediction')
   ax.set_xlabel(r'$n$ (samples)', fontsize=16)
   ax.set_ylabel(r'$x_n$, $\hat{x}_n$', fontsize=16)
   ax.set_xlim([-10,n])
   ax.set_yticks([-1,0,1])
   ax.plot([frame_size, frame_size], [-1,1], '--r')
   ax.legend(loc='upper right')
   pl.tight_layout()



def plot_orig_vs_reconst(model, n=0):
   fig = pl.figure()
   pl.plot(in_frames[n], 'k')
   pl.plot(out_frames[0][n], 'k')
   pl.plot(model.predict(in_frames[n:n+1])[0], 'r')

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((out_frames[0][::step] - lin_model.predict(in_frames[::step])).T, 'k', alpha=0.2)
   gca().tight_layout()



def plot_results(model, encoder, loss_recorder, num=3000):

   pl.figure()
   pl.semilogy(loss_recorder.losses)

   plot_orig_vs_reconst(model, 0)
   plot_prediction(model, num, make_1freq)

   if model == lin_model:
      pl.figure()
      pl.plot(model.get_weights()[0], 'k')
      pl.plot(model.get_weights()[2].T, 'b')
      pl.title("learned weights")

   # TODO: code is a circle even for untrained model --> WHY?
   # TODO learn code from iteration
   predict_gen = lambda n: P.predict_signal2(model, in_frames[0], shift, n)
   pred_frames, *_ = TS.make_training_set(predict_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift)
   code = encoder.predict(pred_frames)
   # code = encoder.predict(in_frames)
   if n_latent == 2:
      pl.figure()
      pl.plot(*code.T, 'k')
   elif n_latent == 3:
      TS.plot3d(*code.T, 'k', linewidth=0.5, alpha=0.9)
      ax = pl.gca()
      ax.set_xlabel(r'$z_1$', fontsize=20)
      ax.set_ylabel(r'$z_2$', fontsize=20)
      ax.set_zlabel(r'$z_3$', fontsize=20)
      pl.tight_layout()

def plot3(data):
   TS.plot3d(*data.T, 'k', linewidth=0.5, alpha=0.9)
   ax = pl.gca()
   ax.set_xlabel(r'$z_1$', fontsize=20)
   ax.set_ylabel(r'$z_2$', fontsize=20)
   ax.set_zlabel(r'$z_3$', fontsize=20)
   pl.tight_layout()


plot_results(lin_model, lin_encoder, lin_loss_recorder)
plot_results(nl_model, nl_encoder, nl_loss_recorder)



from pylab import *
