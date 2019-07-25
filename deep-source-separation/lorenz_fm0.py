# TODO
# • increase receptive field -> try with all signals
# • scan two sinusoids
#   • does it break down when 1/(freq-diff) is smaller then receptive field?
#   • what happens if the main freq/scale is of the order of the receptive field
#   • When it breaks down what helps
#     • increasing receptive field
#     • more samples to improve statistics ?!!??
#       • what if we have phase diffusion?
# • Adversarial WaveNet Source Separation
#   • run two wavenets (from sound to sound) in parallel
#   • critic supports independence


from separator import *

sig1, sig2 = 0.6*lorenz, 0.4*fm_strong0
signal = sig1 + sig2
frame_size = 256

sep = Separator(
   signal=signal(10000),
   # noise_stddev=1.,
   # noise_decay=0.01,
   coder_factory=ConvFactory(
      input_size=frame_size,
      latent_sizes=[4, 4],
      kernel_size=3,
      # features=[4, 4, 4, 4, 4, 4, 4],
      features=[4, 4, 8, 8, 16, 16, 16, 32],
      #features=[4, 8, 8, 8, 16, 32, 32],

      #dec_features=[8, 8, 4, 4, 4, 4, 4],
      dec_features=[8, 8, 8, 8, 8, 8, 8, 8],
      #decoder_noise=.2,
      one_one_conv=True,
      #one_one_conv=True,
      #decoder_noise=dict(stddev=0.8, decay=0.0001, final_stddev=0.05),

      #dec_features=[16, 16, 16, 16, 16],
      #decoder_noise={'dropout': 0.3},

      #resnet=True,
   ),
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(lr=0.005),
   #info_loss=0.5
   #optimizer=keras.optimizers.Adam(lr=0.01),
   #optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001),
   #vanishing_xprojection=True

   # optimizer=keras.optimizers.Adam(0.0001),
   # critic_optimizer=keras.optimizers.Adam(0.0005),
   # adversarial=0.2,
   # critic_runs=10,
)

sep.model.summary()
train_and_summary(sep, 20)
