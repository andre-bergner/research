from separator import *

#sig1, sig2 = 0.3*lorenz, 0.05*fm_strong
sig1, sig2 = 0.1*lorenz, 0.2*fm_strong
signal = sig1 + sig2
power = 7
frame_size, stride = 2 ** power, 2


sep = Separator(
    signal=signal(10000),
    stride=stride,
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[4, 4],
        kernel_size=3,
        upsample_with_zeros=True,
        features=[32]*power,
        activation=leaky_tanh(0),
        decoder_noise=dict(stddev=0.1, decay=0.00001, final_stddev=0.02, correlation_dims='all'),
    ),
    latent_noise=dict(stddev=0.1, decay=0.00001, final_stddev=0.01, correlation_dims=[4, 4]),
    loss='mae',
    signal_gens=[sig1, sig2],
    #optimizer=keras.optimizers.Nadam(lr=.001, beta_1=0.99, beta_2=0.999),
    #optimizer=keras.optimizers.Adam(lr=.001, beta_1=0.99, beta_2=0.999),
    optimizer=keras.optimizers.Nadam(lr=.001, beta_1=0.99, beta_2=0.999, schedule_decay=0),
)


train_and_summary(sep, 100, 16)
train_and_summary(sep, 300, 16)

# train_and_summary(sep, 500, 16)
# train_and_summary(sep, 500, 32)
# train_and_summary(sep, 500, 64)
# train_and_summary(sep, 500, 128)
