from separator import *

sig1, sig2 = 0.3*lorenz, 0.2*fm_strong
signal = sig1 + sig2
frame_size, stride = 64, 2

sep = Separator(
    signal=signal(10000),
    stride=stride,
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[3, 3],
        kernel_size=3,
        upsample_with_zeros=True,
        features=[32]*6,
        #features=[4, 8, 8, 16, 16, 32],
        decoder_noise=dict(stddev=0.1, decay=0.0001, final_stddev=0.01, correlation_dims='all'),
    ),
    loss='mae',
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(.001),
)

train_and_summary(sep, 100, 16)
train_and_summary(sep, 50, 32)
train_and_summary(sep, 50, 64)
