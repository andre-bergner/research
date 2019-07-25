from separator import *

sinmod = LazyGenerator(lambda n: np.sin(np.pi*0.1*np.arange(n) + 20*np.sin(0.00599291*np.arange(n))))
sin2 = make_sin_gen(np.pi*0.1)  # same base frequency as modulated sinusoid
square = LazyGenerator(lambda n: np.tanh(1000*sin2(n)))
sig1, sig2 = square, sinmod

signal = sig1 + sig2

N_FRAMES = 10000
POWER = 6
STRIDE = 1
FRAME_SIZE = 2 ** POWER
LATENT_SPACE = [3, 3]

coder_factory=ConvFactory(
    input_size=FRAME_SIZE,
    latent_sizes=LATENT_SPACE,
    kernel_size=3,
    features=[16] * POWER,
    upsample_with_zeros=True,
    activation=leaky_tanh(0),
    decoder_noise={"stddev": 0.3, "decay": 0.0001, "final_stddev": 0.},
)

def make_sep():
    return Separator(
        signal=signal(N_FRAMES),
        stride=STRIDE,
        coder_factory=coder_factory,
        signal_gens=[sig1, sig2],
        loss='mae',
        #input_noise=dict(stddev=1.5, decay=0.00001, final_stddev=0.01),
        #latent_noise=dict(stddev=0.2, decay=0.00001, final_stddev=0.02, correlation_dims=LATENT_SPACE),
        input_noise={"stddev": 1.0, "decay": 0.0001, "final_stddev": 0.},
        optimizer=keras.optimizers.Adam(0.001),
        #critic_optimizer=keras.optimizers.Adam(0.0005),
        #adversarial=0.2,
        #critic_runs=10,
    )


seps = []
for n in range(10):
    sep = make_sep()
    seps.append(sep)
    print("run {}/10".format(n+1))
    sep.train(10, batch_size=8)
    sep.train(10, batch_size=16)
    training_summary(sep)
