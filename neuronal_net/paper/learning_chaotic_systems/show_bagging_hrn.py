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



frame_size = 120
shift = 8
n_latent = 3
noise_stddev = 0.02
n_pairs = 10000

make_signal = lambda n: TS.hindmarsh_rose3(30*n, [0,0,4], 3.85, -1.744)[::30, 0]

in_frames, *_ = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=4)


activation = fun.bind(XL.tanhx, alpha=0.1)
act = lambda: L.Activation(activation)
dense = lambda s: F.dense(s, activation=None)


def make_dense_model(example_frame, latent_size):
   sig_len = np.size(example_frame)
   x = F.input_like(example_frame)
   eta = F.noise(noise_stddev)

   assert latent_size <= sig_len//4

   enc1 = dense([sig_len//2])  >> act()
   enc2 = dense([sig_len//4])  >> act() >> F.batch_norm()
   enc3 = dense([latent_size]) >> act() >> F.batch_norm()
   dec3 = dense([sig_len//2])  >> act() >> F.batch_norm()
   dec2 = dense([sig_len//4])  >> act() >> F.batch_norm()
   dec1 = dense([sig_len]) 

   encoder = enc1 >> enc2 >> enc3
   decoder = dec3 >> dec2 >> dec1
   y = eta >> encoder >> decoder
   latent = encoder(x)
   out = decoder(latent)

   try:
      dzdx = XL.jacobian(latent,x)
   except:
      dzdx = None
      pass

   return M.Model([x], [y(x)]), M.Model([x], [y(x), y(y(x))]), M.Model([x], [latent]), dzdx


model, model2, encoder, dzdx = make_dense_model(in_frames[0], n_latent)
model.load_weights('../hindmarsh_rose3__120_3.long_train.hdf5')
model.summary()

sig = make_signal(4096)
pred_sig = P.predict_signal2(model, in_frames[380], shift, 4096)


def plot_bagging(n_frames=200, zoom=False, legend_loc='best'):
   start_frame = in_frames[380:381].copy()
   frames = np.array([f[0] for f in P.generate_n_frames_from(model, start_frame, n_frames)])
   times = [np.arange(n, n+frame_size) for n in shift*np.arange(n_frames)]

   frames = []
   frame = start_frame.copy()
   #frame = start_frame.reshape([1] + list(start_frame.shape))
   fad_frames = start_frame[0].copy()
   for n in range(n_frames):
      frame = model.predict(frame)
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
      pl.plot(t, f, color='#558866', alpha=0.4, label=label)
      label = None
   #pl.plot(sig[380+shift:380+shift+len(fad_frames)], linestyle='--', color='k', linewidth=linewidth, label=r'$x_n$ (ground truth)')
   pl.plot(fad_frames, color='blue', linewidth=linewidth, label=r'$\hat{x}_n$ (faded averaged prediction)')
   pl.plot(avg_frames, color='red', linewidth=linewidth, label=r'$\hat{x}_n$ (averaged prediction)')
   if zoom:
      ax.set_xlim([980, 1100])
      ax.set_ylim([-0.06, 0.06])
   else:
      ax.set_xlim([800, 1600])
   ax.set_xlabel(r'$n$ (samples)', fontsize=16)
   ax.set_ylabel(r'$x_n$', fontsize=16)
   pl.legend(loc=legend_loc, fontsize=14)
   pl.tight_layout()

plot_bagging()
plot_bagging(zoom=True)
