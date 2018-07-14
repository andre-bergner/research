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

from timeshift_autoencoder import predictors as P



frame_size = 128
shift = 8
n_pairs = 10000
n_latent1 = 3
n_latent2 = 3
n_epochs = 20
noise_stddev = 0.01


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
   y1 = (eta() >> encoder >> slice1 >> decoder1 >> eta() >> encoder >> slice1 >> decoder1)(x)
   y2 = (eta() >> encoder >> slice2 >> decoder2 >> eta() >> encoder >> slice2 >> decoder2)(x)
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

   return (
      M.Model([x], [loss([x, y])]),
      M.Model([x], [y]),
      None,#M.Model([x], [y, y(y)]),
      M.Model([x], [y1]),
      M.Model([x], [y2]),
      M.Model([x], [encoder(x)]),
      None#dzdx
   )




def make_conv_model(example_frame, latent_sizes=[n_latent1, n_latent2]):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)

   encoder = (  F.reshape([sig_len,1])
             >> F.conv1d(4, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(8, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(8, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(16, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(32, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(32, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.conv1d(32, 5, 2)  >> act() #>> F.batch_norm() # >> F.dropout(0.5)
             >> F.flatten()
             >> dense([latent_size]) >> act()
             )

   slice1 = XL.Slice[:,0:latent_sizes[0]]
   decoder1 = (  dense([1,32]) >> act()
              >> F.up1d() >> F.conv1d(32, 5)  >> act()
              >> F.up1d() >> F.conv1d(32, 5)  >> act()
              >> F.up1d() >> F.conv1d(32, 5)  >> act()
              >> F.up1d() >> F.conv1d(16, 5)  >> act()
              >> F.up1d() >> F.conv1d(8, 5)  >> act()
              >> F.up1d() >> F.conv1d(4, 5)  >> act()
              >> F.up1d() >> F.conv1d(1, 5)
              >> F.flatten()
              )

   slice2 = XL.Slice[:,latent_sizes[0]:]
   decoder2 = (  dense([1,32]) >> act()
              >> F.up1d() >> F.conv1d(32, 5)  >> act()
              >> F.up1d() >> F.conv1d(32, 5)  >> act()
              >> F.up1d() >> F.conv1d(32, 5)  >> act()
              >> F.up1d() >> F.conv1d(16, 5)  >> act()
              >> F.up1d() >> F.conv1d(8, 5)  >> act()
              >> F.up1d() >> F.conv1d(4, 5)  >> act()
              >> F.up1d() >> F.conv1d(1, 5)
              >> F.flatten()
              )

   ex = encoder(x)
   z1 = slice1(ex)
   z2 = slice2(ex)
   y1 = decoder1(z1)
   y2 = decoder2(z2)
   #y1 = (eta() >> encoder >> slice1 >> decoder1 >> eta() >> encoder >> slice1 >> decoder1)(x)
   #y2 = (eta() >> encoder >> slice2 >> decoder2 >> eta() >> encoder >> slice2 >> decoder2)(x)
   #y = lambda x: L.Add()([y1(x), y2(x)])
   y = L.Add()([y1, y2])

   m = M.Model([x], [y])

   m.add_loss(K.mean(K.square((encoder >> slice1 >> decoder1)(y2))))
   m.add_loss(K.mean(K.square((encoder >> slice2 >> decoder2)(y1))))

   return (
      None,
      m, #M.Model([x], [y]),
      None,
      M.Model([x], [y1]),
      M.Model([x], [y2]),
      M.Model([x], [encoder(x)]),
      None
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


#make_2freq = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(np.pi*0.05*np.arange(n))
sin0 = lambda n: 0.3*np.sin(0.03*np.arange(n))
#sin1 = lambda n: 0.64*np.sin(0.05*np.arange(n))
sin2 = lambda n: 0.3*np.sin(np.pi*0.05*np.arange(n))
tanhsin1 = lambda n: 0.6*np.tanh(4*np.sin(0.05*np.arange(n)))
fm_soft = lambda n: np.sin(0.07*np.arange(n) + 4*np.sin(0.00599291*np.arange(n)))
fm_soft1 = lambda n: np.sin(np.pi*0.05*np.arange(n) + 3*np.sin(0.00599291*np.arange(n)))
fm_soft3 = lambda n: np.sin(np.pi*0.1*np.arange(n) + 6*np.sin(0.00599291*np.arange(n)))
fm_soft3inv = lambda n: np.sin(np.pi*0.1*np.arange(n) - 6*np.sin(0.00599291*np.arange(n)))
fm_soft2 = lambda n: np.sin(0.15*np.arange(n) + 18*np.sin(0.00599291*np.arange(n)))
fm_med = lambda n: np.sin(0.1*np.arange(n) + 1*np.sin(0.11*np.arange(n)))
fm_strong = lambda n: 0.5*np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
fm_hyper = lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)) + 2*np.sin(0.009*np.arange(n)))
lorenz = lambda n: TS.lorenz(n, [1,0,0])[::1]
lorenz2 = lambda n: TS.lorenz(n+25000, [0,-1,0])[25000:]

#sig1 = lambda n: 0.3*lorenz(n)
#sig2 = lambda n: 0.3*fm_strong(n)
#sig2 = lambda n: 0.3*fm_soft1(n)
#sig2 = lambda n: 0.3*lorenz2(n)

sig1 = lambda n: 0.3*fm_soft3(n)
sig2 = lambda n: 0.3*fm_soft3inv(n)
make_2freq = lambda n: sig1(n) + sig2(n)

frames, out_frames, *_ = TS.make_training_set(make_2freq, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

frames1, *_ = TS.make_training_set(sig1, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
frames2, *_ = TS.make_training_set(sig2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)




trainer, model, model2, mode1, mode2, encoder, dzdx = make_conv_model(frames[0])
#_, model, model2, mode1, mode2, encoder, encoder2 = make_model2(frames[0])
loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) #+ 0.001*K.sum(dzdx*dzdx)

model.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
model.summary()
plot_model(model, to_file='ssae.png', show_shapes=True)

x = F.input_like(frames[0])
cheater = M.Model([x], [mode1(x), mode2(x)])
cheater.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)

#model2.compile(optimizer=keras.optimizers.Adam(), loss='mse')
#model2.summary()

#trainer.compile(optimizer=keras.optimizers.Adam(0.0001,0.5), loss=lambda y_true, y_pred:y_pred)
#trainer.summary()

#mode1.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
#mode1.summary()



loss_recorder = tools.LossRecorder()

tools.train(model, frames, frames, 128, 200, loss_recorder)
#tools.train(model, frames, out_frames[0], 32, n_epochs, loss_recorder)
#tools.train(model, frames, out_frames[0], 128, 15*n_epochs, loss_recorder)
#tools.train(cheater, frames, [frames1, frames2], 32, n_epochs, loss_recorder)
#tools.train(model2, frames, out_frames, 32, n_epochs, loss_recorder)
#tools.train(trainer, frames, frames, 32, n_epochs, loss_recorder)



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

def plot_joint_dist2():
   code1 = encoder.predict(frames)
   code2 = encoder2.predict(frames)
   code = np.concatenate([code1.T,code2.T]).T
   fig, axs = plt.subplots(n_latent1+n_latent2, n_latent1+n_latent2, figsize=(8, 8))
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.5)
         ax.axis('off')



code = encoder.predict(frames)

plot_modes2()
plot_joint_dist()

# plot(P.predict_signal(model, frames[0], shift, 5000), 'k')
# plot(make_2freq(1000))



def xcorr(m1, m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x2 ) / sqrt( mean(x1**2) * mean(x2**2) )

def nlcorr(m1, m2):
   x1 = m1 - mean(tanh(m1))
   x2 = m2 - mean(tanh(m2))
   return mean( x1 * x2 ) / sqrt( mean(x1**2) * mean(x2**2) )

def xcorr12(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x2 * x2 ) / (sqrt(mean(x1**2)) * mean(x2**3)**(1/3))

def xcorr21(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x1 * x2 ) / (sqrt(mean(x2**2)) * mean(x1**3)**(1/3))

def xcorr22(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x1 * x2 * x2 ) / (sqrt(mean(x2**2)) * mean(x1**2)**(1/2))

def xcorr33(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x1 * x1 * x2 * x2 * x2) / sqrt( mean(x2**2) * mean(x1**2) )

def print_corr(X, fxcorr):
   np.set_printoptions(precision=3, suppress=True)
   print(np.array([[xcorr12(c1, c2) for c1 in code.T] for c2 in code.T]))
