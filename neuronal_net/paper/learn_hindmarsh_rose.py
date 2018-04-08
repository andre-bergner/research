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

from pylab import *

frame_size = 120
shift = 8
n_latent = 3
noise_stddev = 0.02
n_pairs = 10000
n_epochs = 100

loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

make_signal = lambda n: TS.hindmarsh_rose3(30*n, [0,0,4], 3.85, -1.744)[::30]

in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

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
      pass

   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [latent]), None


model, model2, encoder, dhdx = make_dense_model(in_frames[0], n_latent)

model.summary()

loss_recorder = tools.LossRecorder()

model2.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
tools.train(model2, in_frames, out_frames, 32, n_epochs, loss_recorder)
tools.train(model2, in_frames, out_frames, 64, 5*n_epochs, loss_recorder)
tools.train(model2, in_frames, out_frames, 128, 15*n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(), loss=loss_function)


figure()
semilogy(loss_recorder.losses)


def plot_orig_vs_reconst(n=0):
   fig = pl.figure()
   pl.plot(in_frames[n], 'k')
   pl.plot(out_frames[0][n], 'k')
   pl.plot(model.predict(in_frames[n:n+1])[0], 'r')

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((out_frames[0][::step] - model.predict(in_frames[::step])).T, 'k', alpha=0.2)

plot_orig_vs_reconst(0)


def plot_prediction(sig, n=2000):
   pred_sig = P.predict_signal2(model, sig[:frame_size], shift, n+100)
   fig, ax = pl.subplots(2,1)
   ax[0].plot(sig[:n], 'k')
   ax[0].plot(pred_sig[:n], 'r')
   ax[1].plot(sig[:n]-pred_sig[:n])

predict_gen = lambda n: P.predict_signal2(model, in_frames[0], shift, n)
pred_frames, *_ = TS.make_training_set(predict_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift)
code = encoder.predict(pred_frames)

sig = make_signal(5000)

plot_prediction(sig, 3000)

#TS.plot3d(*dot(rot([1,0,1],-1.4),code.T), '-k', linewidth=0.5)
