from separator import *

fixed_sin = make_sin_gen(np.pi*0.1)
scan_sin = make_sin_gen(0.1)
sig1, sig2 = fixed_sin, scan_sin
signal = sig1 + sig2

frame_size = 32 # 128
latent_sizes = [2, 2]

coder_factory=ConvFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   kernel_size=3,
   features=[16]*5,
   upsample_with_zeros=True,
   activation=leaky_tanh(0),   # scaled version 0.1*x
   decoder_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.05, "correlation_dims":'all'},
)

def make_separator():
   return Separator(
      signal=signal(5000),
      coder_factory=coder_factory,
      signal_gens=[sig1, sig2],
      input_noise={"stddev": 1, "decay": 0.0001, "final_stddev": 0.05, "correlation_dims":'all'},
      optimizer=keras.optimizers.Adam(0.001),
   )

# sep = make_separator()
# train_and_summary(sep, 10, 16)

seps = []
for _ in range(20):
   sep = make_separator()
   seps.append(sep)
   train_and_summary(sep, 20, 16)
