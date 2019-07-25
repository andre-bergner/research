from separator import *

fixed_sin = make_sin_gen(np.pi*0.1)
scan_sin = make_sin_gen(0.1)

TODO overlapping modulated sines using adversarial 

sig1, sig2 = fixed_sin, scan_sin

signal = sig1 + sig2

frame_size = 64
latent_sizes = [2, 2]


class LinearFactory:

   def __init__(self, input_size, latent_sizes, decoder_noise=None):
      self.input_size = input_size
      self.latent_sizes = latent_sizes

      if isinstance(decoder_noise, (int, float)):
         self.dec_noise = lambda: F.noise(decoder_noise)
      elif isinstance(decoder_noise, dict):
         self.dec_noise = lambda: F.noise(**decoder_noise)
      else:
         self.dec_noise = lambda: lambda x: x

      latent_size = sum(self.latent_sizes)

   def make_encoder(self, latent_size=None):
      if latent_size == None: latent_size = sum(self.latent_sizes)
      return F.dense([latent_size])

   def make_decoder(self):
      return fun._ >> self.dec_noise() >> F.dense([self.input_size])





#coder_factory = DenseFactory(
#   input_size=frame_size,
#   latent_sizes=latent_sizes,
#   layer_sizes=[]
#)

coder_factory = LinearFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
)

sep = Separator(
   signal=signal(5000),
   coder_factory=coder_factory,
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(0.0001),
   critic_optimizer=keras.optimizers.Adam(0.0005),
   adversarial=0.5,
   critic_runs=10,
)

train_and_summary(sep, 20)

### Run the model from scratch many times:
# seps = []
# for _ in range(10):
#    sep = Separator(
#       signal=signal(5000),
#       noise_stddev=.5,
#       #noise_decay=0.01,
#       coder_factory=coder_factory,
#       signal_gens=[sig1, sig2],
#       optimizer=keras.optimizers.Adam(0.01),
#    )
#    seps.append(sep)
#    train_and_summary(sep, 100)
