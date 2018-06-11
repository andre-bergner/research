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
n_latent2 = 3
n_epochs = 100
noise_stddev = 0.1


activation = fun.bind(XL.tanhx, alpha=0.1)
act = lambda: L.Activation(activation)


def make_model(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)


   encoder = (  F. dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
             >> F. dense([sig_len//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> F. dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             )

   slice1 = F.fun._ >> XL.Slice[:,0:latent_sizes[0]]
   decoder1 = (  slice1
              >> F.dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   slice2 = F.fun._ >> XL.Slice[:,latent_sizes[0]:]
   decoder2 = (  slice2
              >> F.dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   # IDEAS:
   # • constraint on separated channels ? e.g. less fluctuations
   # • SAE with with separation in z-space

   # y = eta() >> encoder >> eta() >> decoder
   y1 = eta() >> encoder >> eta() >> decoder1
   y2 = eta() >> encoder >> eta() >> decoder2
   add = L.Add()
   # latent = encoder(x)
   return M.Model([x], [add([y1(x), y2(x)])]), M.Model([x], [y1(x)]), M.Model([x], [y2(x)])






def make_model2(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)


   encoder1 = (  F. dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
              >> F. dense([sig_len//4])  >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F. dense([latent_sizes[0]]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              )

   decoder1 = (  F.dense([sig_len//4]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   encoder2 = (  F. dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
              >> F. dense([sig_len//4])  >> act() >> F.batch_norm() # >> F.dropout(0.2)
              >> F. dense([latent_sizes[1]]) >> act() >> F.batch_norm() # >> F.dropout(0.2)
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
   add = L.Add()
   # latent = encoder(x)
   return M.Model([x], [add([y1(x), y2(x)])]), M.Model([x], [y1(x)]), M.Model([x], [y2(x)])


def windowed(xs, win_size):
   if win_size <= len(xs):
      for n in range(len(xs)-win_size+1):
         yield xs[n:n+win_size]

def build_prediction(model, num=2000):
   pred_frames = model.predict(frames[:num])
   times = [np.arange(n, n+frame_size) for n in np.arange(len(pred_frames))]
   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, pred_frames):
      avg_frames[t] += f
   avg_frames *= 1./frame_size
   return avg_frames


#make_2freq = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(np.pi*0.05*np.arange(n))
sin1 = lambda n: 0.64*np.sin(0.05*np.arange(n))
tanhsin1 = lambda n: 0.6*np.tanh(4*np.sin(0.05*np.arange(n)))
fm_soft = lambda n: np.sin(0.07*np.arange(n) + 4*np.sin(0.00599291*np.arange(n)))
fm_soft2 = lambda n: np.sin(0.15*np.arange(n) + 18*np.sin(0.00599291*np.arange(n)))
fm_med = lambda n: np.sin(0.1*np.arange(n) + 1*np.sin(0.11*np.arange(n)))
fm_strong = lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
fm_hyper = lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)) + 2*np.sin(0.009*np.arange(n)))
lorenz = lambda n: TS.lorenz(n, [1,0,0])[::1]
lorenz2 = lambda n: TS.lorenz(n, [0,-1,0])[::1]
sig1 = lorenz
sig2 = lambda n: 0.3*np.sin(np.pi*0.05*np.arange(n))
#sig2 = lorenz2
make_2freq = lambda n: sig1(n) + sig2(n)

frames, out_frames, *_ = TS.make_training_set(make_2freq, frame_size=frame_size, n_pairs=n_pairs, shift=shift)




model, mode1, mode2 = make_model(frames[0])
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
model.summary()


loss_recorder = tools.LossRecorder()
#tools.train(model, frames, out_frames[0], 32, n_epochs, loss_recorder)
#tools.train(model, frames, out_frames[0], 128, 15*n_epochs, loss_recorder)
tools.train(model, frames, frames, 32, n_epochs, loss_recorder)
tools.train(model, frames, frames, 128, 15*n_epochs, loss_recorder)



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


plot_modes2()

# plot(P.predict_signal(model, frames[0], shift, 5000), 'k')
# plot(make_2freq(1000))


