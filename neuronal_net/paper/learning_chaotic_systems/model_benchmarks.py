import numpy as np
import scipy.signal as ss

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

import sys
sys.path.append('../../')

from keras_tools import tools
from keras_tools import upsampling as Up
from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS
from keras_tools.wavfile import *

from timeshift_autoencoder import predictors as P

import pylab as pl




frame_size = 128
shift = 8
n_latent = 3
in_noise_stddev = 0.03
code_noise_stddev = 0.03
n_pairs = 10000
n_epochs = 100
resample = 1

signal_gen = lambda n: TS.lorenz(resample*n)[::resample]
signal_gen3 = lambda n: TS.lorenz_all(resample*n)[::resample]
#signal_gen = lambda n: TS.hindmarsh_rose3(20*n, [0,0,4], 3.85, -1.744)[::20,0]



#### experimenting with resampling the signal and scaling the model window accordingly
# resample = 4
# frame_size = 32
# shift = 2

#signal_gen = lambda n: np.sin(0.05*np.arange(n)) + 0.3*np.sin(0.2212*np.arange(n))
#n_latent = 8

#signal_gen = lambda n: np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#signal_gen = lambda n: np.sin(0.8*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#signal_gen = lambda n: tools.add_noise( np.sin(0.1*np.arange(n) + 6*np.sin(0.017*np.arange(n))), 0.1)
#signal_gen = lambda n: np.exp(-0.001*np.arange(n)) * np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#n_pairs = 2000
#n_epochs = 100
#shift = 4

#wav = loadwav("7.wav")[390:,0]
#wav = loadwav("SDHIT04.WAV")[:,0]
#signal_gen = lambda n: wav[:n]
#frame_size = 128
#n_latent = 24
#shift = 3


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

   #def on_train_end(self, logs={}):
   def on_epoch_end(self, epoch, logs={}):
      self.predictions.append(self.predictor())




default_configuration = {
   "frame_size": 128,
   "shift": 8,
   "n_latent": 4,
   "in_noise_stddev": 0.05,
   "code_noise_stddev": 0.01,
}


def models(config=default_configuration):

   activation = fun.bind(XL.tanhx, alpha=0.1)

   act = lambda: L.Activation(activation)
   softmax = lambda: L.Activation(L.activations.softmax)
   eta1 = lambda: F.noise(config["in_noise_stddev"])
   eta2 = lambda: F.noise(config["code_noise_stddev"])

   n_latent = config["n_latent"]

   def arnn_model(example_frame):
      frame_size = example_frame.shape[-1]

      x = F.input_like(example_frame)

      d1 = F.dense([int(frame_size/2)]) >> act()
      d2 = F.dense([int(frame_size/4)]) >> act()
      d3 = F.dense([n_latent]) >> act()
      d4 = F.dense([int(frame_size/2)]) >> act()
      d5 = F.dense([int(frame_size/4)]) >> act()
      d6 = F.dense([1]) >> act()

      chain = eta1() >> d1 >> d2 >> d3 >> eta2() >> d4 >> d5 >> d6

      return M.Model([x], [chain(x)])


   def parnn_model(example_frame, bins=64):
      frame_size = example_frame.shape[-1]

      x = F.input_like(example_frame)

      d1 = F.dense([int(frame_size/2)]) >> act()
      d2 = F.dense([int(frame_size/2)]) >> act()
      d3 = F.dense([int(frame_size/4)]) >> act()
      d4 = F.dense([int(frame_size/4)]) >> act()
      d5 = F.dense([bins]) >> softmax()

      chain = eta1() >> d1 >> d2 >> d3 >> d4 >> d5

      return M.Model([x], [chain(x)])


   def tae_model(example_frame):
      frame_size = np.size(example_frame)
      x = F.input_like(example_frame)

      enc1 = F.dense([int(frame_size/2)]) >> act()
      enc2 = F.dense([int(frame_size/4)]) >> act()
      enc3 = F.dense([n_latent]) >> act()
      dec3 = F.dense([int(frame_size/2)]) >> act()
      dec2 = F.dense([int(frame_size/4)]) >> act()
      dec1 = F.dense([frame_size]) #>> act()

      encoder = enc1 >> enc2 >> enc3
      decoder = dec3 >> dec2 >> dec1
      chain = eta1() >> encoder >> eta2() >> decoder
      latent = encoder(x)
      out = chain(x)

      return M.Model([x], [out]), M.Model([x], [out, chain(out)]), M.Model([x], [latent]), XL.jacobian(latent,x)

   return arnn_model, parnn_model, tae_model




loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

arnn_model, parnn_model, tae_model = models({
   "frame_size": frame_size,
   "shift": shift,
   "n_latent": n_latent,
   "in_noise_stddev": in_noise_stddev,
   "code_noise_stddev": code_noise_stddev,
})



in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(
   signal_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

next_samples_p = np.array([P.x2p(x) for x in next_samples])

make_test_signal = lambda n: TS.lorenz(n, [0,1,0])

def prediction_dist(predictor, num_pred=10, pred_frames=5):
   pred_len = frame_size * pred_frames
   sig = make_test_signal(num_pred*frame_size + pred_len)
   diffs = []
   for n in np.arange(num_pred):
      frame = sig[n*frame_size:(n+1)*frame_size].copy()
      sig_pred = predictor(frame, pred_len)[:pred_len]
      diffs.append(sig_pred - sig[n*frame_size:n*frame_size+pred_len])
   return np.std(np.array(diffs), axis=0)



arnn = arnn_model(in_frames[0])
parnn = parnn_model(in_frames[0])
tae, _, encoder, _ = tae_model(in_frames[0])
tae21, tae22, *_ = tae_model(in_frames[0])

parnn_metrics = Metrics(fun.bind(P.predict_par_model, arnn, start_frame=in_frames[0], n_samples=2048))
arnn_metrics = Metrics(fun.bind(P.predict_ar_model, arnn, start_frame=in_frames[0], n_samples=2048))
tae_metrics = Metrics(fun.bind(P.predict_signal, tae, start_frame=in_frames[0], shift=shift, n_samples=2048))
tae2_metrics = Metrics(fun.bind(P.predict_signal, tae21, start_frame=in_frames[0], shift=shift, n_samples=2048))

#parnn_metrics = Metrics( fun.bind(prediction_dist, predictor=lambda f,x: predict_par_model(parnn, start_frame=f, n_samples=x)))
#arnn_metrics = Metrics(  fun.bind(prediction_dist, predictor=lambda f,x: predict_ar_model(arnn, start_frame=f, n_samples=x)))
#tae_metrics = Metrics(   fun.bind(prediction_dist, predictor=lambda f,x: predict_signal(tae, start_frame=f, shift=shift, n_samples=x)))
#tae2_metrics = Metrics(  fun.bind(prediction_dist, predictor=lambda f,x: predict_signal(tae21, start_frame=f, shift=shift, n_samples=x)))


def train_model(model, ins, outs, metrics_recorder,loss=loss_function):
   model.compile(optimizer=keras.optimizers.Adam(), loss=loss)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   tools.train(model, ins, outs, 128, n_epochs, metrics_recorder)

# train_model(parnn, in_frames, next_samples_p, parnn_metrics, loss=keras.losses.categorical_crossentropy)
# train_model(arnn, in_frames, next_samples, arnn_metrics)
# train_model(tae, in_frames, out_frames[0], tae_metrics)
# train_model(tae22, in_frames, out_frames, tae2_metrics)

# parnn.save_weights('parnn.z3.ver1.hdf5')
# arnn.save_weights('arnn.z3.ver1.hdf5')
# tae.save_weights('tae.z3.ver1.hdf5')
# tae22.save_weights('tae22.z3.ver1.hdf5')
parnn.load_weights('parnn.z3.ver1.hdf5')
arnn.load_weights('arnn.z3.ver1.hdf5')
tae.load_weights('tae.z3.ver1.hdf5')
tae22.load_weights('tae22.z3.ver1.hdf5')

sig = signal_gen(4096)
pred_par_sig = P.predict_par_model(parnn, in_frames[0], 4096)
pred_ar_sig = P.predict_ar_model(arnn, in_frames[0], 4096)
pred_sig = P.predict_signal2(tae, in_frames[0], shift, 4096)
pred_sig2 = P.predict_signal2(tae21, in_frames[0], shift, 4096)

#f0 = np.random.rand(*in_frames[0].shape)
#pred_par_sig = P.predict_par_model(parnn, f0, 4096)
#pred_ar_sig = P.predict_ar_model(arnn, f0, 4096)
#pred_sig = P.predict_signal(tae, f0, shift, 4096)
#pred_sig2 = P.predict_signal(tae21, f0, shift, 4096)


def plot_results():
   fig, ax = pl.subplots(3,2)
   ax[0,0].semilogy(tae_metrics.losses, 'b')
   ax[0,0].semilogy(tae2_metrics.losses, 'g')
   ax[0,0].semilogy(arnn_metrics.losses, 'r')
   ax[0,0].semilogy(parnn_metrics.losses, 'c')

   ax[0,1].plot(sig[:-int(15/resample)], sig[int(15/resample):], 'k', linewidth=0.5)
   ax[1,0].plot(pred_sig[:-int(15/resample)], pred_sig[int(15/resample):], 'b', linewidth=0.5)
   ax[1,1].plot(pred_sig2[:-int(15/resample)], pred_sig2[int(15/resample):], 'g', linewidth=0.5)
   ax[2,0].plot(pred_ar_sig[:-int(15/resample)], pred_ar_sig[int(15/resample):], 'r', linewidth=0.5)
   ax[2,1].plot(pred_par_sig[:-int(15/resample)], pred_par_sig[int(15/resample):], 'c', linewidth=0.5)


def plot_trajectories(n=2000):
   fig = pl.figure(figsize=(8,2))
   ax = pl.gca()
   pl.plot(sig[:n], 'k', label='groud truth')
   pl.plot(pred_sig[:n], 'b', label='SAE')
   pl.plot(pred_ar_sig[:n], 'r', label='ARNN')
   pl.plot(pred_par_sig[:n], 'c', label='PARNN')
   ax.set_xlabel(r'$n$ (samples)', fontsize=16)
   ax.set_ylabel(r'$x_n$, $\hat{x}_n$', fontsize=16)
   ax.set_xlim([-10,n])
   ax.set_yticks([-1,0,1])
   ax.plot([frame_size, frame_size], [-1,1], '--r')
   ax.legend(loc='upper right')
   pl.tight_layout()


plot_results()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import matplotlib.patches as patches

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def plot_tae_results(n=2500):
   sig3 = signal_gen3(3000)[1000:]

   fig = pl.figure(figsize=(10,2))
   ax1 = fig.add_axes([0.07, 0.3, 0.45, 0.6])
   ax2 = fig.add_axes([0.58, 0.15, 0.18, 0.75], projection='3d')
   ax3 = fig.add_axes([0.78, 0.15, 0.18, 0.75], projection='3d')
   fig.text(0.02, 0.85, 'a)', fontsize=16)
   fig.text(0.57, 0.85, 'b)', fontsize=16)
   fig.text(0.76, 0.85, 'c)', fontsize=16)

   ax1.plot(sig[:n], color='k', label='groud truth')
   ax1.plot(pred_sig[:n], color='dodgerblue', label='SAE')
   ax1.set_xlabel(r'$n$ (samples)', fontsize=16)
   ax1.set_ylabel(r'$x_n$, $\hat{x}_n$', fontsize=16)
   ax1.set_xlim([-10,n])
   ax1.set_yticks([-1,0,1])
   ax1.plot([frame_size, frame_size], [-1,1], '--r')
   ax1.legend(loc='upper right')

   ax2.plot(*sig3.T, 'k', linewidth=0.5)
   ax2.view_init(13, -45)
   ax2.set_xticklabels([])
   ax2.set_yticklabels([])
   ax2.set_zticklabels([])
   ax2.set_xlabel(r'$u_1$', fontsize=14)
   ax2.set_ylabel(r'$u_2$', fontsize=14)
   ax2.set_zlabel(r'$u_3$', fontsize=14)
   ax2.xaxis.labelpad = -12
   ax2.yaxis.labelpad = -12
   ax2.zaxis.labelpad = -12

   predict_gen = lambda n: P.predict_signal2(tae, in_frames[0], shift, n)
   pred_frames, *_ = TS.make_training_set(predict_gen, frame_size=frame_size, n_pairs=5000, shift=shift)
   code = encoder.predict(pred_frames)
   ax3.plot(*code.T, color='dodgerblue', linewidth=0.5)
   ax3.view_init(30, -25)
   ax3.set_xticklabels([])
   ax3.set_yticklabels([])
   ax3.set_zticklabels([])
   ax3.set_xlabel(r'$z_1$', fontsize=14)
   ax3.set_ylabel(r'$z_2$', fontsize=14)
   ax3.set_zlabel(r'$z_3$', fontsize=14)
   ax3.xaxis.labelpad = -12
   ax3.yaxis.labelpad = -12
   ax3.zaxis.labelpad = -12

plot_tae_results()
