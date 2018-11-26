from imports import *
from coder_factories import *
from keras_tools import test_signals as TS


class Separator:

   def __init__(self
   ,  signal, frame_size
   ,  latent_sizes
   ,  noise_stddev=2.0
   ,  noise_decay=0.002
   ,  factory=ConvFactory  # DenseFactory, DilatedFactory
   ,  verbose=False
   ):

      assert(type(frame_size) == int)
      assert(type(latent_sizes) == list or type(latent_sizes) == tuple)

      FRAME_SIZE0 = 128
      #FRAME_SIZE0 = 256
      #FRAME_SIZE0 = 64
      factor = frame_size // FRAME_SIZE0

      # factory = ConvFactory2
      # frame_size = 132
      # factor = 1

      self.frames = np.array([w for w in windowed(signal, frame_size, 1)])

      self.model, self.encoder, _, self.modes = make_factor_model(
         self.frames[0], factory(
            self.frames[0], latent_sizes, use_batch_norm=False, scale_factor=factor,
            kernel_size=5,
            #features=[2, 2, 2, 4, 4, 4, 8, 8]
            features=[4, 4, 4, 8, 8, 8, 16, 16]
            #features=[4, 4, 4, 8, 8, 8, 16]
            #features=[4, 4, 8, 16, 16, 32, 32]
         ),
         noise_stddev=noise_stddev, noise_decay=noise_decay, shared_encoder=True
      )

      #self.model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001), loss='mse')
      self.model.compile(optimizer=keras.optimizers.Adam(lr=0.002), loss='mse')

      if verbose:
         self.model.summary()
         plot_model(self.model, to_file='ssae.png', show_shapes=True)

      self.loss_recorder = LossRecorder(lambda n,m: pred_error(self.modes[n], self.frames, [sig1,sig2][m], 2048))



   def train(self, n_epochs):

      tools.train(self.model, self.frames, self.frames, 128, n_epochs, self.loss_recorder)
      pred = build_prediction(self.model, self.frames)



def separate(signal, frame_size, latent_sizes):
   separator = Separator(signal, frame_size, latent_sizes)
   try:
      separator.train(20)
   except:
      print("Error while training network!")
      pass
   return separator


# TODO: two sweeps of different frequencies
#sig1, sig2 = lorenz, 0.8*fm_strong
#sig1, sig2 = lorenz, 0.5*fm_strong0
#sig1, sig2 = lorenz, 0.1*fm_strong0
#sig1, sig2 = 0.3*lorenz, 0.15*fm_strong
#sig1, sig2 = lorenz, 0.5*fm_strong
#sig1, sig2 = 0.3*lorenz, 0.15*fm_strong0
sig1, sig2 = 0.3*lorenz, 0.2*fm_strong0
sig_gen = sig1 + sig2
#signal = sig_gen(5000)
#signal = sig_gen(2200)
signal = sig_gen(10000)

sep = separate(signal, 256, [4,4])
training_summary(sep.model, *sep.modes, sep.encoder, sig_gen, sig1, sig2, sep.frames, sep.loss_recorder)


def plot_modes3(sep, n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(sep.modes[0], sep.frames, n), 'r')
   plot(build_prediction(sep.modes[1], sep.frames, n), 'r')

# import theano.d3viz as d3v
# d3v.d3viz(loss, 'loss.html')
