# ANALYSIS PLAN
# • setup no-noise, fix batch-order?
# • measure success for different parameters (frame-size, )
# • record init & weights for several runs → analyse success in dependence of these

# TODO
# try with noisy input 

# TODO
# • generalize to N-channels
# • train 3 sin on 3 channels
# • train 2 sin on 3 channels
# • train 3 sin on 2 channels

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


frame_size = 128
n_pairs = 5000
n_latent = [2, 2, 2]
n_epochs = 30
noise_stddev = 0.0#5


activation = fun.bind(XL.tanhx, alpha=0.2)
# activation = None
act = lambda: L.Activation(activation)
# act = lambda: L.LeakyReLU(alpha=0.2)
# OBSERVATION: LeakyReLU is less successful (or not all)


# OBSERVATION: ingredients for reproducable successful separation:
# • large frame sizes: 32 was almost always failing, 128 works (almost) always
# • enforcing projector property further helps to stabilize training
# • noise does not seem to have big impact (at least in simple setup)
# • nonliearity is important, training is unstable otherwise

dense = F.dense

def make_model(example_frame, latent_sizes=n_latent):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)

   encoder = (  #dense([sig_len//2])  >> act() >>
                dense([latent_size]) >> act()
             )

   encoder1 = dense([latent_sizes[0]]) >> act()
   encoder2 = dense([latent_sizes[1]]) >> act()

   latent_sizes2 = np.concatenate([[0], np.cumsum(latent_sizes)])

   def make_decoder(n):
      slice = fun._ >> XL.Slice[ :, latent_sizes2[n]:latent_sizes2[n+1] ]
      decoder = (  #dense([sig_len//2]) >> act() >>
                   dense([sig_len]) >> act()
                )
      return slice >> decoder

   # constraint: separators are contractive (+noise) projectors (re-encode single channel):
   # NEXT STEP interleave y-outs and train against them as well

   decoders = [make_decoder(n) for n in range(len(latent_sizes))]
   #channels = [(eta() >> encoder >> dec)(x) for dec in decoders]
   channels = [(eta() >> encoder >> dec >> eta() >> encoder >> dec)(x) for dec in decoders]
   #channels = [(eta() >> enc >> dec)(x) for enc,dec in zip(encoders, decoders)]
   y = L.Add()(channels)

   m = M.Model([x], [y])
   # m.add_loss(1*K.mean(K.square((encoder >> decoder1)(y2))))
   # m.add_loss(1*K.mean(K.square((encoder >> decoder2)(y1))))

   return (
      m,
      M.Model([x], [encoder(x)]),
      [M.Model([x], [c]) for c in channels]
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
signal_gen = lambda n: tools.add_noise(sig1(n) + sig2(n), 0.1)

frames = np.array([w for w in windowed(signal_gen(n_pairs+frame_size), frame_size, 1)])

model, encoder, modes = make_model(frames[0])
mode1 = modes[0]
mode2 = modes[1]
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
   for m in modes:
      plot(build_prediction(m, n), 'r')

def plot_modes2(n=2000):
   figure()
   plot(build_prediction(mode1, n))
   plot(build_prediction(mode2, n))

def plot_orig_vs_reconst(n=2000):
   plot(build_prediction(model, n))
   plot(signal_gen(n))

def plot_joint_dist():
   code = encoder.predict(frames)
   z_dim = sum(n_latent)
   fig, axs = plt.subplots(z_dim, z_dim, figsize=(8, 8))
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.5)
         ax.axis('off')

code = encoder.predict(frames)

plot_modes3()
plot_joint_dist()
