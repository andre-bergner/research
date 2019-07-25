from separator import *

sin1 = LazyGenerator(lambda n: np.sin(np.pi*0.1*np.arange(n) + 20*np.sin(0.00599291*np.arange(n))))
sin2 = make_sin_gen(np.pi*0.1)
square = LazyGenerator(lambda n: np.tanh(1000*sin2(n)))
sig1, sig2 = square, sin1

signal = sig1 + sig2

N_FRAMES = 10000
POWER = 6
STRIDE = 1
FRAME_SIZE = 2 ** POWER
LATENT_SPACE = [2, 3]

coder_factory=ConvFactory(
    input_size=FRAME_SIZE,
    latent_sizes=LATENT_SPACE,
    kernel_size=3,
    features=[8, 2, 2, 4, 8, 16],
    upsample_with_zeros=True,
    activation=leaky_tanh(0),
    #decoder_noise={"stddev": 0.8, "decay": 0.0001, "final_stddev": 0.05, "correlation_dims":'all'},
)

sep = Separator(
    signal=signal(N_FRAMES),
    stride=STRIDE,
    coder_factory=coder_factory,
    signal_gens=[sig1, sig2],
    loss='mae',
    input_noise=dict(stddev=1.5, decay=0.0001, final_stddev=0.01),
    latent_noise=dict(stddev=0.1, decay=0.0001, final_stddev=0.01, correlation_dims=LATENT_SPACE),
    #optimizer=keras.optimizers.Adam(0.001),
    optimizer=keras.optimizers.Adam(0.0005),
)

train_and_summary(sep, 20, 8)
train_and_summary(sep, 30, 16)
train_and_summary(sep, 50, 16)
