from separator import *

sig1, sig2 = kicks_sin1
signal = sig1 + sig2
frame_size = 128

FEATURES, LEARNING_RATE, BATCH_SIZE = 16, 0.001, 128
#FEATURES, LEARNING_RATE, BATCH_SIZE = 32, 0.0002, 16
#FEATURES, LEARNING_RATE, BATCH_SIZE = 16, 0.0002, 1   # BATCH_SIZE=1???

sep = Separator(
   signal=signal(10000),
   stride=2,
   coder_factory=ConvFactory(
      input_size=frame_size,
      latent_sizes=[4, 4],
      kernel_size=3,
      upsample_with_zeros=True,
      features=[FEATURES] * 7,
      activation=leaky_tanh(0),
      decoder_noise={"stddev": .2, "decay": 0.0001, "final_stddev": 0.},
   ),
   input_noise={"stddev": .5, "decay": 0.001, "final_stddev": 0.},
   loss='mae',
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
)

sep.model.summary()
train_and_summary(sep, 25, BATCH_SIZE)

# train_and_summary(sep, 100, 16)
# train_and_summary(sep, 900, 32)
