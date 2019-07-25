# TODO
# • fix distance measure (norm on length and variance)
# • move predictor into separator
# • factor out critic

from separator import *

#sig1, sig2 = down_sample(0.3*lorenz, 2), down_sample(0.2*fm_strong, 2)
sig1, sig2 = 0.3*lorenz, 0.15*fm_strong
signal = sig1 + sig2
frame_size = 128

sep = Separator(
   signal=signal(30000),
   coder_factory=ConvFactory(
      input_size=frame_size,
      latent_sizes=[3, 3],
      kernel_size=3,
      features=[4, 8, 8, 16, 16, 16, 32],
      dec_features=[16, 16, 8, 8, 4, 4, 2],
      decoder_noise=dict(stddev=0.2, decay=0.01, final_stddev=0.),
   ),
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(lr=.01, decay=0.),
)

train_and_summary(sep, 10)
