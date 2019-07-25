print("""
    --------------------------------------------------------
    LORENZ + RÖSSLER
    
    This example  demonstrates a successful  separation of a
    Lorenz and a Rössler time series.
    Both signals have a continuous spectrum with overlapping
    strong components.
    --------------------------------------------------------
""")

from separator import *

sig1, sig2 = 0.5*lorenz, 1.0*rossler
signal = sig1 + sig2
POWER = 7
FRAME_SIZE = 2 ** POWER
STRIDE = 2

sep = Separator(
    signal=signal(20000),
    stride=STRIDE,
    coder_factory=ConvFactory(
        input_size=FRAME_SIZE,
        latent_sizes=[3, 3],
        features=[32] * POWER,
        kernel_size=3,
        upsample_with_zeros=True,
        activation=leaky_tanh(0.),
        #decoder_noise={"stddev": 0.2, "decay": 0.0001, "final_stddev": 0.01, "correlation_dims":'all'},
    ),
    loss='mae',
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.001),
)


NUM = 3
WIDTH = 1/NUM - 0.02
fig = figure(figsize=(10,3))
for epoch in range(NUM):
    sep.train(n_epochs=1, batch_size=32)
    sep.model.save_weights('movie_lor-ros_epoch_{:02}.hdf5'.format(epoch))
    x_pos = epoch/NUM
    plot_latent_space_impl(
        fig, [x_pos, 0.05, x_pos + WIDTH, 0.95], sep.frames, sep.encoder, with_labels=False,
        max_num_frames=None, method='hist2', bins=70)

#train_and_summary(sep, n_epochs=1, batch_size=16)
#train_and_summary(sep, n_epochs=1, batch_size=16)
#train_and_summary(sep, n_epochs=1, batch_size=16)
# train_and_summary(sep, n_epochs=20, batch_size=4)
# train_and_summary(sep, n_epochs=20, batch_size=8)
# train_and_summary(sep, n_epochs=20, batch_size=16)
# train_and_summary(sep, n_epochs=40, batch_size=32)
# #train_and_summary(sep, n_epochs=25, batch_size=16)
# #train_and_summary(sep, n_epochs=25, batch_size=32)
