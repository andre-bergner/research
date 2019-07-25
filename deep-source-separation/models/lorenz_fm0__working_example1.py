from separator import *

sig1, sig2 = 0.6*lorenz, 0.4*fm_strong0
signal = sig1 + sig2
frame_size = 256

sep = Separator(
   signal=signal(10000),
   coder_factory=ConvFactory(
      input_size=frame_size,
      latent_sizes=[6, 6],   # TODO [4,4]
      kernel_size=3,
      features=[4, 4, 8, 8, 16, 16, 16, 32],
      dec_features=[8, 8, 8, 8, 8, 8, 8, 8],
      decoder_noise=.1,
      one_one_conv=True,
   ),
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(lr=0.002),
   info_loss=0.1
)

sep.model.summary()
train_and_summary(sep, 50)
