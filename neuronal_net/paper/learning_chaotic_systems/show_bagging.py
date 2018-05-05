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
n_latent = 8
in_noise_stddev = 0.03
code_noise_stddev = 0.03
n_pairs = 10000
n_epochs = 100
resample = 1

signal_gen = lambda n: TS.lorenz(resample*n)[::resample]



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

   return tae_model



def make_nonlinear_model(example_frame, latent_size):
   act = lambda: L.Activation(fun.bind(XL.tanhx, alpha=0.1))
   frame_size = np.size(example_frame)
   x = F.input_like(example_frame)

   eta = F.noise(in_noise_stddev)

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


loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

tae_model = models({
   "frame_size": frame_size,
   "shift": shift,
   "n_latent": n_latent,
   "in_noise_stddev": in_noise_stddev,
   "code_noise_stddev": code_noise_stddev,
})

signal_gen = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(np.pi*0.05*np.arange(n))

in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(
   signal_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

#tae, *_ = tae_model(in_frames[0])
tae, *_ = make_nonlinear_model(in_frames[0], 3)
tae.compile(optimizer=keras.optimizers.Adam(), loss='mae')
#tae.load_weights('tae.ver2.hdf5')
#tae.load_weights('../nl_n128_s8_zdim3__long_train.ver2.hdf5')
tae.load_weights('../nl_n128_s8_zdim3.80_epochs.hdf5')

sig = signal_gen(4096)
pred_sig = P.predict_signal(tae, in_frames[0], shift, 4096)


def plot_bagging(n_frames=120, zoom=False, legend_loc='best'):
   start_frame = in_frames[0:1].copy()
   frames = np.array([f[0] for f in P.generate_n_frames_from(tae, in_frames[0:1], n_frames)])
   times = [np.arange(n, n+frame_size) for n in shift*np.arange(n_frames)]
   #avg_frames = np.zeros(times[-1][-1]+1)
   #for t, f in zip(times, frames):
   #   avg_frames[t] += f

   #avg_frames = start_frame[0].copy()
   #for f in frames[0:]:
   #   avg_frames = P.xfade_append(avg_frames.T, f.T, shift).T
   #avg_frames = avg_frames[shift:]

   frames = []
   frame = start_frame.copy()
   #frame = start_frame.reshape([1] + list(start_frame.shape))
   fad_frames = start_frame[0].copy()
   for n in range(n_frames):
      frame = tae.predict(frame)
      fad_frames = P.xfade_append(fad_frames, frame[0], shift)
      frame = fad_frames[-frame_size:].reshape([1] + list(start_frame[0].shape))
      frames.append(frame[0])
   fad_frames = fad_frames[shift:]

   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, frames):
      avg_frames[t] += f
   avg_frames *= shift/frame_size

   con_frames = np.concatenate([ start_frame[0], *[f[-shift:] for f in frames] ])

   linewidth = 1 if not zoom else 2
   pl.figure(figsize=(8,3))
   ax = pl.gca()
   label = r'$\mathbf{\hat{\Xi}}$ (predicted frames)'
   for t, f in zip(times, frames):
      pl.plot(t, f, color='dodgerblue', alpha=0.4, label=label)
      label = None
   pl.plot(sig[shift:shift+len(fad_frames)], linestyle='--', color='k', linewidth=linewidth, label=r'$x_n$ (ground truth)')
   pl.plot(fad_frames, color='blue', linewidth=linewidth, label=r'$\hat{x}_n$ (faded averaged prediction)')
   pl.plot(avg_frames, color='red', linewidth=linewidth, label=r'$\hat{x}_n$ (averaged prediction)')
   if zoom:
      #ax.set_xlim([350, 450])
      #ax.set_ylim([-0.1, 0.4])
      ax.set_xlim([610, 760])
      ax.set_ylim([-0.9, 0.5])
   else:
      ax.set_xlim([400, 1000])
   ax.set_xlabel(r'$n$ (samples)', fontsize=16)
   ax.set_ylabel(r'$x_n$', fontsize=16)
   pl.legend(loc=legend_loc, fontsize=14)
   pl.tight_layout()

   return frames, avg_frames, con_frames

plot_bagging(legend_loc='lower left')
plot_bagging(zoom=True)
