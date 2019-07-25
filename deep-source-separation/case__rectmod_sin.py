from separator import *

#sin1 = make_sin_gen(np.pi*0.1)
sin1 = LazyGenerator(lambda n: np.sin(np.pi*0.1*np.arange(n) + 15*np.sin(0.00599291*np.arange(n))))
sin2 = make_sin_gen(0.1)
sin1, sin2 = sin2, sin1
rect = LazyGenerator(lambda n: np.tanh(1000*sin2(n)))
sig1, sig2 = rect, sin1

signal = sig1 + sig2

frame_size = 64
latent_sizes = [2, 3]

coder_factory=ConvFactory(
    input_size=frame_size,
    latent_sizes=latent_sizes,
    kernel_size=3,
    features=[4, 4, 4, 8, 8, 16],
    decoder_noise={"stddev": .5, "decay": 0.0005, "final_stddev": 0.01, "correlation_dims":'all'},
    # the correlated noise significantly improves separation and convergence!
)

sep = Separator(
    signal=signal(5000),
    coder_factory=coder_factory,
    signal_gens=[sig1, sig2],
    loss='mae',
    optimizer=keras.optimizers.Adam(0.001),
)

sep.model.summary()
train_and_summary(sep, 20, 16)
#train_and_summary(sep, 30, 32)
