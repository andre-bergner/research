from separator import *

sig1, sig2 = kicks_sin1
signal = sig1 + sig2
frame_size = 128

# sep = Separator(
#    signal=signal(5000),
#    coder_factory=ConvFactory(
#       input_size=frame_size,
#       latent_sizes=[4, 4],
#       kernel_size=3,
#       upsample_with_zeros=True,
#       # features=[4, 4, 4, 4, 4, 4, 4],
#       # features=[4, 4, 8, 8, 16, 16, 16],
#       # dec_features=[8, 8, 8, 8, 8, 8, 8],
#       features=[4, 4, 8, 8, 16, 16, 16, 32],
#       dec_features=[8, 8, 8, 8, 8, 8, 8, 8],
#       # features=[4, 4, 8, 8, 8, 16, 16, 32, 32],
#       # dec_features=[16, 16, 16, 16, 16, 16, 16, 16, 16],
#       # decoder_noise=.1,
#       # decoder_noise={'dropout': 0.1},
#       # one_one_conv=True,
#    ),
#    signal_gens=[sig1, sig2],
#    optimizer=keras.optimizers.Adam(lr=0.001),
#    #info_loss=0.1
# )

# TODO try to learn just the kick, i.e. no separation, just embedding

sep = Separator(
   signal=signal(10000),
   stride=2,
   coder_factory=ConvFactory(
      input_size=frame_size,
      latent_sizes=[4, 4],
      kernel_size=3,
      upsample_with_zeros=True,
      features=[32] * 7,
      activation=leaky_tanh(0),
      #decoder_noise=.1,
      decoder_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.01, "correlation_dims":'all'},
   ),
   loss='mae',
   #latent_noise=0.05,
   #info_loss=0.05,
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(lr=0.0002),
   # critic_optimizer=keras.optimizers.Adam(0.000),
   # adversarial=0.1,
   # critic_runs=10,
)

sep.model.summary()
train_and_summary(sep, 50, 16)
