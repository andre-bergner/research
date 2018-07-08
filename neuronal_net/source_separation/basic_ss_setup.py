import numpy as np

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


frame_size = 80
shift = 8
n_pairs = 5000
n_latent1 = 2
n_latent2 = 2
n_epochs = 30
noise_stddev = 0.05


activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)
#act = lambda: L.LeakyReLU(alpha=0.2)

dense = F.dense

def make_model(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)

   encoder = (  dense([sig_len//2])  >> act()
             >> dense([latent_size]) >> act()
             )

   slice1 = XL.Slice[:,0:latent_sizes[0]]
   decoder1 = (  dense([sig_len//2]) >> act()
              >> dense([sig_len]) 
              )

   slice2 = XL.Slice[:,latent_sizes[0]:]
   decoder2 = (  dense([sig_len//2]) >> act()
              >> dense([sig_len]) 
              )

   ex = (eta() >> encoder >> eta())(x)
   z1 = slice1(ex)
   z2 = slice2(ex)
   y1 = decoder1(z1)
   y2 = decoder2(z2)
   y = L.Add()([y1, y2])

   return (
      M.Model([x], [y]),
      M.Model([x], [y1]),
      M.Model([x], [y2]),
      M.Model([x], [encoder(x)]),
   )



def windowed(xs, win_size, hop=None):
   if hop == None: hop = win_size
   if win_size <= len(xs):
      for n in range(0, len(xs)-win_size+1, hop):
         yield xs[n:n+win_size]

def build_prediction(model, num=2000):
   pred_frames = model.predict(frames[:num])
   times = [np.arange(n, n+frame_size) for n in np.arange(len(pred_frames))]
   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, pred_frames):
      avg_frames[t] += f
   avg_frames *= 1./frame_size
   return avg_frames


sin1 = lambda n: 0.64*np.sin(0.05*np.arange(n))
sin2 = lambda n: 0.3*np.sin(np.pi*0.05*np.arange(n))
sig1 = sin1
sig2 = sin2
make_2freq = lambda n: sig1(n) + sig2(n)

frames, out_frames, *_ = TS.make_training_set(make_2freq, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

model, mode1, mode2, encoder = make_model(frames[0])
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
model.summary()

loss_recorder = tools.LossRecorder()
tools.train(model, frames, frames, 32, n_epochs, loss_recorder)


from pylab import *

figure()
semilogy(loss_recorder.losses)


def plot_modes(n=0):
   figure()
   plot(frames[n], 'k')
   plot(model.predict(frames[n:n+1])[0], 'b')
   plot(mode1.predict(frames[n:n+1])[0], 'r')
   plot(mode2.predict(frames[n:n+1])[0], 'r')
   plot(sig1(frame_size+n)[n:n+frame_size], 'g')
   plot(sig2(frame_size+n)[n:n+frame_size], 'g')


def plot_modes3(n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(mode1, n), 'r')
   plot(build_prediction(mode2, n), 'r')

def plot_modes2(n=2000):
   figure()
   plot(build_prediction(mode1, n))
   plot(build_prediction(mode2, n))

def plot_orig_vs_reconst(n=2000):
   plot(build_prediction(model, n))
   plot(make_2freq(n))

def plot_joint_dist():
   code = encoder.predict(frames)
   fig, axs = plt.subplots(n_latent1+n_latent2, n_latent1+n_latent2, figsize=(8, 8))
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.5)
         ax.axis('off')

code = encoder.predict(frames)

plot_modes2()
plot_joint_dist()
