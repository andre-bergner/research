#  TODO
#  • DenseFactory -> configurable layer sizes
#  • Factories -> decoder-noise parameter
#  • input_noise = number | {"init": x, "decay": y, "target": z }
# --------------------------------------------
#  • normalized distance
#  • log d-loss, log d-prediction
#  • run three models: vanilla, adves, noisy
#  • latent space plot: show scale on outer axes
# --------------------------------------------
#  • pre-train critic on arbitrary factored latent spaces?


from separator import *

sig1, sig2 = 0.3*lorenz, 0.2*fm_strong
signal = sig1 + sig2

frame_size = 128
latent_sizes = [8, 8]

# TODO pass in
# • both optimizers

coder_factory=DenseFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   decoder_noise=None,
)

sep = Separator(
   noise_stddev=0.05,
   noise_decay=0.001,
   signal=signal(5000),
   coder_factory=coder_factory,
   signal_gens=[sig1, sig2],
   adversarial=0.5,
   critic_runs=50
)

train_and_summary(sep, 20)
