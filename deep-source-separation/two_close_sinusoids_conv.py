#  TODO
#  • try adding more layers to latent space encoder

from separator import *

sig1, sig2 = make_sin_gen(0.1), make_sin_gen(0.1111)

signal = sig1 + sig2

frame_size = 128#*4
latent_sizes = [2, 2]

coder_factory=ConvFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   kernel_size=3,
   features=[4, 4, 4, 8, 8, 8, 16, 16],
   #features=[2, 4, 4, 8, 8, 8, 16, 16],
   #scale_factor=4,
   decoder_noise=0.1
)

sep = Separator(
   signal=signal(10000),
   coder_factory=coder_factory,
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(0.001),
)

train_and_summary(sep, 5)


seps = []
for _ in range(10):
   sep = Separator(signal=signal(10000), coder_factory=coder_factory, signal_gens=[sig1, sig2], optimizer=keras.optimizers.Adam(0.001))
   seps.append(sep)
   train_and_summary(sep, 20)
