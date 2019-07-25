from separator import *

fixed_sin = make_sin_gen(np.pi*0.1)
scan_sin = make_sin_gen(0.1)
sig1, sig2 = fixed_sin, scan_sin
signal = sig1 + sig2

frame_size = 128
latent_sizes = [2, 2]

coder_factory=BlockFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   kernel_size=3,
   # features=[4, 4, 4, 8, 8, 8, 16],
   # dec_features=[8, 4, 4, 4, 2, 2],
   # decoder_noise=0.3
)

sep = Separator(
   signal=signal(5000),
   coder_factory=coder_factory,
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(0.002),
)

train_and_summary(sep, 20)


# seps = []
# for _ in range(10):
#    sep = Separator(
#       signal=signal(5000), 
#       coder_factory=coder_factory, 
#       signal_gens=[sig1, sig2], 
#       optimizer=keras.optimizers.Adam(0.001)
#    )
#    seps.append(sep)
#    train_and_summary(sep, 20)
