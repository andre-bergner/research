from separator import *

sig1, sig2 = 0.3*make_sin_gen(np.pi*0.1), 0.3*fm_soft3
signal = sig1 + sig2

frame_size = 128
latent_sizes = [3, 2]

coder_factory=DenseFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   decoder_noise=0.05,
   vae=False
)

sep = Separator(
   signal=signal(5000),
   coder_factory=coder_factory,
   signal_gens=[sig1, sig2],
   adversarial=0.8,
   critic_runs=30
)

train_and_summary(sep, 20)
