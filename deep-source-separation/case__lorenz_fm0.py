print("""
    --------------------------------------------------------
    LORENZ + LOW-FREQ-FM
    
    This example  demonstrates a successful  separation of a
    Lorenz time series  and a FM-signal - both  signals with
    with a strong  overlap of  their main  components in the
    frequency domain.
    This example only seems to work successfully with a high
    embedding dimension (here 256).
    --------------------------------------------------------
""")

from separator import *

sig1, sig2 = 0.6*lorenz, 0.4*fm_strong0
#sig1, sig2 = 0.2*lorenz, 0.4*fm_strong0
signal = sig1 + sig2
frame_size = 128

sep = Separator(
    signal=signal(20000),
    stride=2,
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[4, 4],
        features=[32] * 7,
        kernel_size=3,                # TODO make this the default value, adapt other scripts
        upsample_with_zeros=True,     # TODO make this the default value, adapt other scripts
        activation=leaky_tanh(0),     # TODO make this the default value, adapt other scripts
        #decoder_noise={"stddev": 0.2, "decay": 0.0001, "correlation_dims": 'all'},
    ),
    loss='mae',
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.001),
)

# sep.model.summary()
#train_and_summary(sep, n_epochs=25, batch_size=16)
train_and_summary(sep, n_epochs=50, batch_size=16)
#train_and_summary(sep, n_epochs=50, batch_size=32)

# feat=16:
#train_and_summary(sep, n_epochs=400, batch_size=32)
#train_and_summary(sep, n_epochs=400, batch_size=64)





# version 1
# lr = 0.04, dec_noise = 0.
# n_epochs=200, batch_size=64
# n_epochs=200, batch_size=128
# n_epochs=200, batch_size=256
# n_epochs=100, batch_size=512

# version 2
# lr = 0.02, dec_noise = 0.1
# n_epochs=200, batch_size=64
# n_epochs=400, batch_size=128
# n_epochs=..., batch_size=256
# n_epochs=..., batch_size=512

# version 3
# lr = 0.02, latent_noise = 0.1
# n_epochs=500, batch_size=64
# n_epochs=400, batch_size=128
# n_epochs=300, batch_size=256
# n_epochs=300, batch_size=512
# ...

# version 3
# lr = 0.02, latent_noise = 0.01, loss = mae
# n_epochs=300, batch_size=64
# n_epochs=200, batch_size=128
# n_epochs=200, batch_size=256
# ...

# version 4
# lr = 0.01, latent_noise = 0.01, loss = mae
# n_epochs=400, batch_size=32
# n_epochs=400, batch_size=64
# ...

# more tests
# • slow lr, decoder noise with decay
# • slow lr, decoder noise, more features
