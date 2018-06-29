import os
import sys
sys.path.append('../')

import numpy as np

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

from keras_tools import tools
from keras_tools import functional as fun
from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import test_signals as TS

from timeshift_autoencoder import predictors as P



def make_model(example_frame, latent_sizes):

   activation = fun.bind(XL.tanhx, alpha=0.2)
   act = lambda: L.Activation(activation)

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   latent_size = sum(latent_sizes)

   encoder = (  F.dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
             >> F.dense([sig_len//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             >> F.dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
             )

   slice1 = XL.Slice[:,0:latent_sizes[0]]
   decoder1 = (  F.fun._ >> slice1
              >> F.dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   slice2 = XL.Slice[:,latent_sizes[0]:]
   decoder2 = (  F.fun._ >> slice2
              >> F.dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
              >> F.dense([sig_len]) 
              )

   #y1 = eta() >> encoder >> eta() >> decoder1
   #y2 = eta() >> encoder >> eta() >> decoder2
   y1 = encoder >> decoder1
   y2 = encoder >> decoder2
   z1 = encoder >> slice1
   z2 = encoder >> slice2
   y = lambda x: L.Add()([y1(x), y2(x)])

   return (
      M.Model([x], [y(x)]),
      M.Model([x], [y1(x)]),
      M.Model([x], [y2(x)]),
      M.Model([x], [z1(x)]),
      M.Model([x], [z2(x)]),
      M.Model([x], [encoder(x)]),
   )



class AdversarialSourceSeparator():

   def __init__(self, signal_generator, latent_sizes, model_factory=make_model):

      self.n_epochs = 100
      self.frame_size = 80
      n_pairs = 5000
      self.latent_sizes = latent_sizes
      self.frames, *_ = TS.make_training_set(signal_generator, frame_size=self.frame_size, n_pairs=n_pairs)

      optimizer = keras.optimizers.Adam() #(0.0002, 0.5)
      activation = fun.bind(XL.tanhx, alpha=0.2)
      act = lambda: L.Activation(activation)
      eta = lambda: F.noise(noise_stddev)

      sig_len = self.frame_size
      x = F.input_like(self.frames[0])

      latent_size = sum(self.latent_sizes)

      encoder = (  F.dense([sig_len//2])  >> act()                   # >> F.dropout(0.2)
                >> F.dense([sig_len//4])  >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                >> F.dense([latent_size]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                )

      slice1 = XL.Slice[:,0:latent_sizes[0]]
      slice2 = XL.Slice[:,latent_sizes[0]:]

      decoder1 = (  F.dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                 >> F.dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                 >> F.dense([sig_len]) 
                 )

      decoder2 = (  F.dense([sig_len//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                 >> F.dense([sig_len//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                 >> F.dense([sig_len]) 
                 )

      z1 = encoder >> slice1
      z2 = encoder >> slice2
      y1 = z1 >> decoder1
      y2 = z2 >> decoder2
      y = lambda x: L.Add()([y1(x), y2(x)])



      z1i = L.Input(shape=(self.latent_sizes[0],))
      transfer_12 = (  F.dense([self.frame_size//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                    >> F.dense([self.frame_size//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                    >> F.dense([self.frame_size]) 
                    )


      z2i = L.Input(shape=(self.latent_sizes[1],))
      transfer_21 = (  F.dense([self.frame_size//4]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                    >> F.dense([self.frame_size//2]) >> act() #>> F.batch_norm() # >> F.dropout(0.2)
                    >> F.dense([self.frame_size]) 
                    )

      self.transfer = M.Model([z1i, z2i], [transfer_12(z1i), transfer_21(z2i)])
      self.transfer.compile(optimizer=optimizer, loss='mse')
      self.transfer.summary()

      static_z1 = L.Input(shape=(self.latent_sizes[0],))
      static_transfer_12 = M.Model([static_z1], [transfer_12(static_z1)])
      static_transfer_12.trainable = False
      static_z2 = L.Input(shape=(self.latent_sizes[1],))
      static_transfer_21 = M.Model([static_z2], [transfer_21(static_z2)])
      static_transfer_21.trainable = False
      # self.static_transfer = M.Model([static_z1, static_z2], [transfer_12(static_z1), transfer_21(static_z2)])
      # self.static_transfer.trainable = False

      self.encoder = M.Model([x], [encoder(x)])
      self.z1      = M.Model([x], [z1(x)])
      self.z2      = M.Model([x], [z2(x)])
      self.y1      = M.Model([x], [y1(x)])
      self.y2      = M.Model([x], [y2(x)])
      self.model   = M.Model([x], [y(x)])

      x = F.input_like(self.frames[0])

      
      def loss_f(args):
         y_true, y_pred = args
         l2 = K.mean(K.square(y_true - y_pred))
         l2_cross21 = K.mean(K.square(y1(x) - static_transfer_21(z2(x)) ))
         l2_cross12 = K.mean(K.square(y2(x) - static_transfer_12(z1(x)) ))
         l2_cross21 = K.exp(-l2_cross21)
         l2_cross12 = K.exp(-l2_cross12)

         def xcorr(a, b):
            Y1 = a - K.mean(a)
            Y2 = b - K.mean(b)
            return K.square( K.mean(Y1 * Y2) / (K.mean(K.square(Y1)) * K.mean(K.square(Y2))) )

         return l2 #+ 0.0001*(xcorr(y1(x), static_transfer_21(z2(x))) + xcorr(y2(x), static_transfer_12(z1(x))))
         #return l2 + 0.01*(l2_cross21 + l2_cross12)# + 0.25*(K.mean(K.square(y1(x))) + K.mean(K.square(y2(x))))

      loss = L.Lambda(loss_f, output_shape=(1,))

      self.trainer = M.Model([x], [loss([x, y(x)])])


      # models = model_factory(self.frames[0], latent_sizes)
      # self.model, self.y1, self.y2, self.z1, self.z2, self.encoder = models

      #self.model.compile(optimizer=optimizer, loss='mse')
      #self.model.summary()
      self.trainer.compile(optimizer=optimizer, loss=lambda y_true, y_pred:y_pred)
      self.trainer.summary()






      self.loss_recorder = tools.LossRecorder()


      """
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.sampler = ResultSampler(self.discriminator, self.generator, self.data.visualize)
      """


   def train(self, epochs, batch_size=32, sample_interval=50):

      #tools.train(self.model, self.frames, self.frames, batch_size, self.n_epochs, self.loss_recorder)

      def train_separator():
         idx = np.random.randint(0, self.frames.shape[0], batch_size)
         batch = self.frames[idx]
         #return self.model.train_on_batch(batch, batch)
         w0 = self.transfer.get_weights()[1].copy()
         loss = self.trainer.train_on_batch(batch, batch)
         assert( np.array_equal(w0, self.transfer.get_weights()[1]) )
         return loss

      def train_correlator():
         idx = np.random.randint(0, self.frames.shape[0], batch_size)
         batch = self.frames[idx]
         z1 = self.z1.predict(batch)
         z2 = self.z2.predict(batch)
         y1 = self.y1.predict(batch)
         y2 = self.y2.predict(batch)
         return self.transfer.train_on_batch([z1, z2], [y2, y1])

      for epoch in range(epochs):
         s_loss = train_separator()
         c_loss = train_correlator()
         #c_loss = train_correlator()

         if (epoch % 100 == 0):
            print('# {:>5} Sep. loss: {:.4f}   Cor. loss: {:.4f}, {:.4f}'.format(epoch, s_loss, c_loss[0], c_loss[1]))

         # if (epoch+1) % sample_interval == 0:
         #    self.sampler.sample_images(epoch+1)
         #    self.sampler.sample_model_weights(epoch+1)




# --------------------------------------------------------------------------------------------------


from pylab import *

def build_prediction(model, frames, num=2000):
   frame_size = len(frames[0])
   pred_frames = model.predict(frames[:num])
   times = [np.arange(n, n+frame_size) for n in np.arange(len(pred_frames))]
   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, pred_frames):
      avg_frames[t] += f
   avg_frames *= 1./frame_size
   return avg_frames

def plot_modes3(n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(ass.y1, ass.frames, n), 'r')
   plot(build_prediction(ass.y2, ass.frames, n), 'r')


def windowed(xs, win_size, hop=None):
   if hop == None: hop = win_size
   if win_size <= len(xs):
      for n in range(0, len(xs)-win_size+1, hop):
         yield xs[n:n+win_size]

def spectrogram(signal, N=256, overlap=0.25):
   hop = int(overlap * N)
   def cos_win(x):
      return x * (0.5 - 0.5*cos(linspace(0,2*pi,len(x))))
   return np.array([ np.abs(np.fft.fft(cos_win(win))[:N//2]) for win in windowed(signal, N, hop) ])

def spec(signal, N=256, overlap=0.25):
   s = spectrogram(signal, N, overlap)
   imshow(log(0.001 + s.T[::-1]), aspect='auto')


if __name__ == '__main__':

   fm_soft1 = lambda n: np.sin(np.pi*0.1*np.arange(n) + 6*np.sin(0.00599291*np.arange(n)))
   fm_soft1inv = lambda n: np.sin(np.pi*0.1*np.arange(n) - 6*np.sin(0.00599291*np.arange(n)))
   lorenz = lambda n: TS.lorenz(n, [1,0,0])[::1]
   fm_strong = lambda n: 0.5*np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)))
   sig1 = lambda n: 0.3*lorenz(n)
   sig2 = lambda n: 0.3*fm_strong(n)
   #sig1 = lambda n: 0.3*fm_soft1(n)
   #sig2 = lambda n: 0.3*fm_soft1inv(n)
   signal = lambda n: sig1(n) + sig2(n)

   ass = AdversarialSourceSeparator(signal, [3, 3])
   #ass.train(epochs=2, batch_size=32, sample_interval=200)
   ass.train(epochs=2000, batch_size=32, sample_interval=200)

   plot(build_prediction(ass.y1, ass.frames))
   plot(build_prediction(ass.y2, ass.frames))

# code = ass.encoder.predict(ass.frames)
# y1 = ass.transfer.predict(code)
# code.shape
# y12 = ass.transfer.predict([code[:,:3], code[:,3:]])
# plot(y12[0])
# from pylab import *
# plot(y12[0])
# plot(y12[0])
# plot(y12[0][0])
# plot(ass.y1.predict(ass.frames[0:1]))
# ass.y1.predict(ass.frames[0:1]).shape
# plot(ass.y1.predict(ass.frames[0:1])[0])
# plot(y12[0][0])
# plot(ass.y2.predict(ass.frames[0:1])[0])