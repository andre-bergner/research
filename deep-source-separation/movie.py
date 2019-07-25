from separator import *

sig1, sig2 = 0.6*lorenz, 0.4*fm_strong0
signal = sig1 + sig2
frame_size = 128

def make_sep():
    return Separator(
        signal=signal(20000),
        stride=2,
        coder_factory=ConvFactory(
            input_size=frame_size,
            latent_sizes=[3, 3],
            features=[32] * 7,
            kernel_size=3,
            upsample_with_zeros=True,
            activation=leaky_tanh(0),
        ),
        loss='mae',
        signal_gens=[sig1, sig2],
        optimizer=keras.optimizers.Adam(lr=0.001),
    )


TRAINING = False

if TRAINING:
    sep = make_sep()
    for epoch in range(100):
        sep.train(n_epochs=1, batch_size=32)
        sep.model.save_weights('movie_lor-fm0_epoch_{:02}.hdf5'.format(epoch))

    training_summary(sep, method='hist2', max_num_frames=None, bins=70) 

seps = []

seps.append(make_sep())
seps[-1].model.load_weights('./paper_models/movie_lor-fm0_epoch_00.hdf5')
seps.append(make_sep())
seps[-1].model.load_weights('./paper_models/movie_lor-fm0_epoch_20.hdf5')
seps.append(make_sep())
seps[-1].model.load_weights('./paper_models/movie_lor-fm0_epoch_40.hdf5')
seps.append(make_sep())
seps[-1].model.load_weights('./paper_models/movie_lor-fm0_epoch_60.hdf5')
seps.append(make_sep())
seps[-1].model.load_weights('./paper_models/movie_lor-fm0_epoch_80.hdf5')


NUM = len(seps)
WIDTH = 1/NUM - 0.02

fig = figure(figsize=(10, 2))

for n, sep in enumerate(seps):
    x_pos = n / NUM
    plot_latent_space_impl(
        fig, [0.01+x_pos, 0.05, 0.01+x_pos + WIDTH, 0.93], sep.frames, sep.encoder,
        with_labels=False, with_marginals=False,
        max_num_frames=None, method='hist2', bins=70)

over_ax = fig.add_axes([0, 0, 1, 1])
over_ax.patch.set_alpha(0)
over_ax.axis('off')

for x in linspace(0, 1, NUM+1)[1:-1]:
    over_ax.plot([x, x], [0.05, 0.95], '-k')
over_ax.set_xlim([0,1])
over_ax.set_ylim([0,1])