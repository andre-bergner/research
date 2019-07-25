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

N_FRAMES = 20000
sig1, sig2 = 0.6*lorenz, 0.4*fm_strong0
#sig1, sig2 = 0.2*lorenz, 0.4*fm_strong0
signal = sig1 + sig2
FRAME_SIZE = 128

sep = Separator(
    signal=signal(20000),
    stride=2,
    coder_factory=ConvFactory(
        input_size=FRAME_SIZE,
        latent_sizes=[4, 4],
        features=[32] * 7,
        kernel_size=3,                # TODO make this the default value, adapt other scripts
        upsample_with_zeros=True,     # TODO make this the default value, adapt other scripts
        activation=leaky_tanh(0),     # TODO make this the default value, adapt other scripts
        decoder_noise={"stddev": 0.2, "decay": 0.0001, "correlation_dims": 'all'},
    ),
    loss='mae',
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.001),
)

sep.model.load_weights('./paper_models/lorenz+fm0_weights_1300.hdf5')
training_summary(sep)



def plot_side_by_side(separator):

    fig = plt.figure(figsize=(14, 2))

    m1, m2 = sep.modes[0].infer(N_FRAMES), sep.modes[1].infer(N_FRAMES)
    s1, s2 = sep.signal_gens[0](N_FRAMES), sep.signal_gens[1](N_FRAMES)

    S1 = abs(fft(s1))[:N_FRAMES//2]
    S2 = abs(fft(s2))[:N_FRAMES//2]
    M1 = abs(fft(m1))[:N_FRAMES//2]
    M2 = abs(fft(m2))[:N_FRAMES//2]

    ax = fig.add_axes([0.04, 0.13, 0.48, 0.8])
    plot(m1, '-', linewidth=1, color='#4466bb', label='separated')
    plot(m2, '-', linewidth=1, color='#44bb66', label='separated')
    plot(s1, '--k', linewidth=1, color='#000000', label='ground truth')
    plot(s2, '--k', linewidth=1, color='#000000')
    ax.set_xlim([1000, 2000])
    ax.set_xticks([1000, 2000])
    ax.set_xlabel('time [samples]', fontsize=12, labelpad=-10)

    def loglog_filled(xs, s, color, label=None):
        num = len(xs)
        loglog(xs, linestyle=s, linewidth=1, color=color, label=label)
        fill_between(range(num), zeros(num), xs, alpha=0.6, color=color)

    ax = fig.add_axes([0.58, 0.13, 0.38, 0.8])
    loglog_filled(M1, '-', color='#4466bb', label='separated')
    loglog_filled(M2, '-', color='#44bb66', label='separated')
    loglog(S1, '--', linewidth=1, color='#000000', label='ground truth')
    loglog(S2, '--', linewidth=1, color='#000000')
    ax.set_xlim([1, N_FRAMES//2])
    ax.set_xticks([1, N_FRAMES//2])
    ax.set_xticklabels(['0', r'$\pi$'])
    ax.set_xlabel('frequency [radians]', fontsize=12, labelpad=-10)

    legend(loc=(0.8, 0.5), framealpha=1)



def plot_modes2(separator):
    # stacked version

    fig = plt.figure(figsize=(8, 4))

    m1, m2 = sep.modes[0].infer(N_FRAMES), sep.modes[1].infer(N_FRAMES)
    s1, s2 = sep.signal_gens[0](N_FRAMES), sep.signal_gens[1](N_FRAMES)

    S1 = abs(fft(s1))[:N_FRAMES//2]
    S2 = abs(fft(s2))[:N_FRAMES//2]
    M1 = abs(fft(m1))[:N_FRAMES//2]
    M2 = abs(fft(m2))[:N_FRAMES//2]

    ax = fig.add_axes([0.07, 0.57, 0.9, 0.4])
    plot(m1[:2000], '-', linewidth=1, color='#4488aa', label='separated')
    plot(m2[:2000], '-', linewidth=1, color='#4488aa', label='separated')
    plot(s1[:2000], '--k', linewidth=0.5, color='#442200', label='original')
    plot(s2[:2000], '--k', linewidth=0.5, color='#442200', label='original')
    xlim([0, 2000])


    def loglog_filled(xs, s, color, label):
        num = len(xs)
        loglog(xs, linestyle=s, linewidth=0.5, color=color, label=label)
        fill_between(range(num), zeros(num), xs, alpha=0.5, color=color)

    ax = fig.add_axes([0.07, 0.07, 0.9, 0.42])
    loglog_filled(M1, '-', color='#4488aa', label='separated')
    loglog_filled(M2, '-', color='#4488aa', label='separated')
    loglog_filled(S1, '--', color='#442200', label='original')
    loglog_filled(S2, '--', color='#442200', label='original')
    xlim([1, N_FRAMES//2])

    legend(loc=(0.82, 0.4), framealpha=1)


def plot_latent_space_for_paper(sep):

    fig = figure(figsize=(4,4))

    plot_latent_space_impl(
        fig, [0.1, 0.1, 0.95, 0.95], sep.frames, sep.encoder,
        method='hist2', max_num_frames=None, bins=80, pow=.6)

    fig.text(0.85, 0.89, r'$p(z_n)$', fontsize=16)

