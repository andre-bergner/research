# forked from source_separation_autoencoder.py
# main goal/changes:
# • implement slow feature analysis to force separation
# • changes: model needs current and next frame as inputs

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
n_latent1 = 3
n_latent2 = 3
n_epochs = 30
noise_stddev = 0.01


activation = fun.bind(XL.tanhx, alpha=0.2)
act = lambda: L.Activation(activation)

dense = F.dense


def make_model(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x1 = F.input_like(example_frame)
   x2 = F.input_like(example_frame)
   x3 = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)


   encoder = (  dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
             >> dense([sig_len//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             )

   slice1 = XL.Slice[:,0:latent_sizes[0]]
   #slice1.activity_regularizer = lambda x: 1. / (10. + K.mean(K.square((x))))
   #slice1.activity_regularizer = lambda x: K.exp(-K.mean(K.square((x))))
   decoder1 = (  F.fun._ >> slice1
              >> dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len]) 
              )

   slice2 = XL.Slice[:,latent_sizes[0]:]
   decoder2 = (  F.fun._ >> slice2
              >> dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> dense([sig_len]) 
              )

   #y1 = eta() >> encoder >> eta() >> decoder1
   #y2 = eta() >> encoder >> eta() >> decoder2
   y1 = encoder >> decoder1
   y2 = encoder >> decoder2
   z1 = encoder >> slice1
   z2 = encoder >> slice2
   y = lambda x: L.Add()([y1(x1), y2(x1)])

   def loss_f(args):
      l2 = K.mean(K.square(x1 - y(x1)))
      loss = l2

      Y1 = y1(x1) - K.repeat_elements(K.mean(y1(x1), axis=1, keepdims=True), 80, axis=1)
      Y2 = y2(x1) - K.repeat_elements(K.mean(y2(x1), axis=1, keepdims=True), 80, axis=1)
      #decorr = K.square( K.mean(Y1 * Y2) / (K.mean(K.square(Y1)) * K.mean(K.square(Y2))) )
      decorr = K.square(K.batch_dot(Y1, K.transpose(Y2) ))

      #z_smoothness = K.sum( K.square(encoder(x2) - encoder(x1)) / ( .0000001 + K.square(encoder(x2) + encoder(x1)) ))
      #z_smoothness = K.square(encoder(x2) - encoder(x1))[:,0]# / ( .0000001 + K.square(encoder(vx2) + encoder(vx1)) )[:,0]
      # z1_smoothness = K.mean(K.square(z1(x2) - z1(x1))) / K.mean(K.square(z1(x1) - K.mean(z1(x1))))
      # z2_smoothness = K.mean(K.square(z2(x2) - z2(x1))) / K.mean(K.square(z2(x1) - K.mean(z2(x1))))
      # z_smoothness = z1_smoothness# + z2_smoothness

      zx1 = encoder(x1)
      zx2 = encoder(x2)
      zx3 = encoder(x3)

      z_smoothness = 1 - K.mean(K.square(zx1*zx2 + zx2*zx3) / ((zx1*zx1 + zx2*zx2) * (zx2*zx2 + zx3*zx3)))


      #s = K.square(encoder(vx2) + encoder(vx1))
      #K.sum( (d[:,0] + d[:,1] + d[:,2]) / (s[:,0] + s[:,1] + s[:,2]) )
      #z_smoothness = K.abs(d[:,0] / s[:,0])  +  K.abs(d[:,1] / s[:,1])  +  K.abs(d[:,2] / s[:,2])
      #y_smoothness = K.mean( K.square(y1(x2) - y1(x1)) )  +  K.mean( K.square(y2(x2) - y2(x1)) )
      #y_smoothness = K.mean( K.square(y1(x1)[1:] - y1(x1)[:-1]) )  +  K.mean( K.square(y2(x1)[1:] - y2(x1)[:-1]) )

      #decorr_z = K.square(K.sum( K.sum(K.tanh(2*z1(x1)), axis=1) * K.sum(K.tanh(2*z2(x1)), axis=1), axis=0))
      def xcor(a,b):
         return K.square(K.sum( K.sum(a-K.mean(a,axis=0), axis=1) * K.sum(b-K.mean(b,axis=0), axis=1), axis=0))
      z1_ = z1(x1)
      z2_ = z2(x1)
      decorr_z = xcor(z1_, z2_)
      decorr_z += xcor(z1_, K.square(z2_))
      decorr_z += xcor(K.square(z1_), z2_)
      decorr_z += xcor(K.square(z1_), K.square(z2_))

      def cov_z2(z1, z2): return K.sum(K.square(z1)) * K.sum(K.square(z2))
      #return l2 + 0.1 * K.square( K.sum(z1(x1)) * K.sum(z2(x1)) ) + 0.1 * (cov_z2(z1(x1), z2(x1)))
      #return l2 + cov_z2(z1_, z2_) #+ decorr_z #+ 0.1 * z_smoothness#+ cov_z2(z1(x2), z2(x1)) + cov_z2(z1(x1), z2(x2)) + cov_z2(z1(x2), z2(x2)))
      #return l2 + z_smoothness #+ decorr #+ y_smoothness 

      az1 = K.sum(K.square(z1_))
      az2 = K.sum(K.square(z2_))
      hollowness = (K.square(az1) + K.square(az2))  - 15*(az1 + az2)

      #loss += decorr
      loss += decorr_z
      #loss += cov_z2(z1_, z2_)
      #loss += 5*hollowness
      loss += z_smoothness
      return loss

   loss = L.Lambda(loss_f, output_shape=(1,))

   return (
      M.Model([x1, x2, x3], [loss([x1, x2, x3, y(x1)])]),
      M.Model([x1], [y(x1)]),
      # -----------------------
      M.Model([x1], [y1(x1)]),
      M.Model([x1], [y2(x1)]),
      M.Model([x1], [encoder(x1)])
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
tanhsin1 = lambda n: 0.6*np.tanh(4*np.sin(0.05*np.arange(n)))
fm_soft = lambda n: np.sin(0.07*np.arange(n) + 4*np.sin(0.00599291*np.arange(n)))
fm_soft1 = lambda n: np.sin(np.pi*0.05*np.arange(n) + 3*np.sin(0.00599291*np.arange(n)))
fm_soft1 = lambda n: np.sin(np.pi*0.1*np.arange(n) + 6*np.sin(0.00599291*np.arange(n)))
fm_soft1inv = lambda n: np.sin(np.pi*0.1*np.arange(n) - 6*np.sin(0.00659291*np.arange(n)))
fm_soft2 = lambda n: np.sin(0.15*np.arange(n) + 18*np.sin(0.00599291*np.arange(n)))
fm_med = lambda n: np.sin(0.1*np.arange(n) + 1*np.sin(0.11*np.arange(n)))
fm_strong = lambda n: 0.5*np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
fm_hyper = lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)) + 2*np.sin(0.009*np.arange(n)))
lorenz = lambda n: TS.lorenz(n, [1,0,0])[::1]
lorenz2 = lambda n: TS.lorenz(n+15000, [0,-1,0])[15000:]
sig1 = lambda n: 0.3*lorenz(n)
sig2 = lambda n: 0.3*fm_strong(n)
#sig2 = lambda n: 0.3*sin2(n)
#sig1 = lambda n: 0.3*fm_soft1(n)
#sig2 = lambda n: 0.3*fm_soft1inv(n)
make_2freq = lambda n: sig1(n) + sig2(n)

frames, out_frames, *_ = TS.make_training_set(make_2freq, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

frames1, *_ = TS.make_training_set(sig1, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
frames2, *_ = TS.make_training_set(sig2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)




trainer, model, mode1, mode2, encoder = make_model(frames[0])

trainer.compile(optimizer=keras.optimizers.Adam(), loss=lambda y_true, y_pred:y_pred)
#trainer.compile(optimizer=keras.optimizers.Adam(0.0001,0.5), loss=lambda y_true, y_pred:y_pred)
trainer.summary()

loss_recorder = tools.LossRecorder()
tools.train(trainer, [frames[:-2], frames[1:-1], frames[2:]], frames[:-2], 32, n_epochs, loss_recorder)



from pylab import *

figure()
semilogy(loss_recorder.losses)


def spectrogram(signal, N=256, overlap=0.25):
   hop = int(overlap * N)
   def cos_win(x):
      return x * (0.5 - 0.5*cos(linspace(0,2*pi,len(x))))
   return np.array([ np.abs(np.fft.fft(cos_win(win))[:N//2]) for win in windowed(signal, N, hop) ])

def spec(signal, N=256, overlap=0.25):
   s = spectrogram(signal, N, overlap)
   imshow(log(0.001 + s.T[::-1]), aspect='auto')

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








def sample_entropy(U, m, r):

   def maxdist(win_n, win_m):
      #result = max([abs(ua - va) for ua, va in zip(win_n, win_m)])
      #return result
      return norm(win_n - win_m)

   def phi(m):
      wins = np.array([[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)])
      C = [
         len([
            1 for j in range(len(wins))
            if i != j and maxdist(wins[i], wins[j]) <= r
         ])
         for i in range(len(wins))
      ]
      return sum(C)

   N = len(U)
    
   return -np.log(phi(m+1) / phi(m))
