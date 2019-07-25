from separator import *

fixed_sin = make_sin_gen(np.pi*0.1)
scan_sin = make_sin_gen(0.1)
sig1, sig2 = fixed_sin, scan_sin

signal = sig1 + sig2

frame_size = 4
latent_sizes = [2, 2]

class ShallowFactory:

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
      return F.dense([latent_size], use_bias=False) >> L.Activation('tanh')

   def make_decoder(self, custom_features=None, custom_noise=None):
      return fun._ >> self.dec_noise() >> F.dense([self.input_size], use_bias=False)


coder_factory = ShallowFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   decoder_noise={"stddev": 1.5, "decay": 0.0001, "final_stddev": 0.01, "correlation_dims":'all'},
   #decoder_noise={"stddev": 1.5, "decay": 0.0001, "final_stddev": 0.01},
   #decoder_noise=0.5
)

def make_sep():
   return Separator(
      signal=signal(20000),
      stride=8,
      coder_factory=coder_factory,
      signal_gens=[sig1, sig2],
      optimizer=keras.optimizers.Nadam(0.001),
   )

# sep = make_sep()
# train_and_summary(sep, 50, 16)

### Run the model from scratch many times:
seps = []
for _ in range(10):
   sep = make_sep()
   seps.append(sep)
   train_and_summary(sep, 50, 16)
