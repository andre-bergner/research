from separator import *

sig1, sig2 = 0.5*lorenz, 1.0*rossler
signal = sig1 + sig2
POWER = 7
FRAME_SIZE = 2 ** POWER
STRIDE = 2

sep = Separator(
    signal=signal(40000),
    stride=STRIDE,
    coder_factory=ConvFactory(
        input_size=FRAME_SIZE,
        latent_sizes=[4, 4],
        features=[32] * POWER,
        kernel_size=3,
        upsample_with_zeros=True,
        activation=leaky_tanh(0.),
    ),
    loss='mae',
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.001),
)

TRAINING = False

if TRAINING:
    train_and_summary(sep, n_epochs=100, batch_size=16)
    train_and_summary(sep, n_epochs=300, batch_size=32)
    train_and_summary(sep, n_epochs=400, batch_size=64)
    train_and_summary(sep, n_epochs=300, batch_size=128)
else:
    sep.model.load_weights("lorenz_rossler_z[4,4]_1100_epochs.hdf5")
    sep.sep_recorder.mutual_information = np.loadtxt("lorenz_rossler_z[4,4]_1100_epochs__MI.txt")


def plot_for_paper(sep):

    fig = figure(figsize=(4,6))

    plot_latent_space_impl(
        fig, [0.1, 0.35, 0.95, 0.95], sep.frames, sep.encoder,
        method='hist2', max_num_frames=None, bins=80, pow=0.8)

    ax = fig.add_axes([0.1, 0.05, 0.85, 0.2])

    src1, src2 = sep.signal_gens[0](2000), sep.signal_gens[1](2000)
    mode1, mode2 = sep.modes[0].infer(2000), sep.modes[1].infer(2000)
    ax.plot(src1, '--k', label='ground truth')
    ax.plot(src2, '--k')
    ax.plot(mode1, '-', color='#4466bb', label='separated')
    ax.plot(mode2, '-', color='#44bb66')
    ax.set_xlim([1100, 1500])
    #ax.xaxis.set_ticks([])
    #ax.yaxis.set_ticks([])
    legend()

    fig.text(0.87, 0.91, r'$p(z_n)$', fontsize=16)


#plot_for_paper(sep)




def plot_latent_space_for_paper(sep):

    fig = figure(figsize=(4,4))

    plot_latent_space_impl(
        fig, [0.1, 0.1, 0.95, 0.95], sep.frames, sep.encoder,
        method='hist2', max_num_frames=None, bins=80, pow=1)

    fig.text(0.85, 0.89, r'$p(z_n)$', fontsize=16)


plot_latent_space_for_paper(sep)
