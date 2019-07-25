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
        input_noise={"stddev": 1.0, "decay": 0.0001, "final_stddev": 0.},
        optimizer=keras.optimizers.Adam(0.001),
    )

sep = make_sep()
#sep.model.load_weights("paper_models/square_sinmod_samefreq__40epochs(8,16,32,32).hdf5")
sep.model.load_weights("paper_models/square-sinmod-samefreq__lr=0.0002_longrun@500.hdf5")



def plot_the_plot(separator):

    plt.rcParams['mathtext.fontset'] = 'stix'

    fig = plt.figure(figsize=(14, 2))

    m1, m2 = sep.modes[0].infer(N_FRAMES), sep.modes[1].infer(N_FRAMES)
    s1, s2 = sep.signal_gens[0](N_FRAMES), sep.signal_gens[1](N_FRAMES)

    FFT = 256
    HOP = 16
    NUM = 8000

    def spec_img(x):
        spec =  np.rot90(spectrogram(x, N=FFT, overlap=HOP/FFT)) #** 0.5
        return (spec - spec.min()) / (spec.max() - spec.min())

    def plot_spec(ax, img):
        img = ax.imshow(img, cmap='gray_r', aspect='auto', extent=[0, 1, 0, 1])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r'$0$', r'$8000$'], fontsize=11)
        ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
        ax.xaxis.get_majorticklabels()[1].set_horizontalalignment('right')
        ax.set_xlabel('time [samples]', labelpad=-11)
        return img

    #box_props = dict(boxstyle='round', facecolor='black', alpha=0.1)
    #box_props = dict(facecolor='black', alpha=0.03)
    box_props = dict(facecolor='white', alpha=0.5)

    ax = fig.add_axes([0.04, 0.13, 0.2, 0.8])
    plot_spec(ax, spec_img((s1+s2)[:NUM]))
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r'$0$', r'$\pi$'], fontsize=11)
    ax.set_ylabel('frequency [radians]')
    ax.text(0.05, 0.82, 'mixture', fontsize=12, bbox=box_props)

    ax = fig.add_axes([0.25, 0.13, 0.2, 0.8])
    plot_spec(ax, spec_img(m1[:NUM]))
    ax.set_yticks([])
    ax.text(0.05, 0.82, 'separated source 1', fontsize=12, bbox=box_props)

    ax = fig.add_axes([0.46, 0.13, 0.2, 0.8])
    img = plot_spec(ax, spec_img(m2[:NUM]))
    ax.set_yticks([])
    ax.text(0.05, 0.82, 'separated source 2', fontsize=12, bbox=box_props)

    ax = fig.add_axes([0.67, 0.13, 0.02, 0.8])
    cb = colorbar(img, cax=ax, cmap='gray_r', orientation='vertical')
    #cb.set_label('loudness (normalized)')

    ax = fig.add_axes([0.73, 0.13, 0.25, 0.8])
    ax.plot(s1, '--k', linewidth=1, label='ground truth')
    ax.plot(s2, '--k', linewidth=1)
    ax.plot(m1, '-', linewidth=1, color='#4466bb', label='separated')
    ax.plot(m2, '-', linewidth=1, color='#44bb66', label='separated')
    ax.set_xlim([400, 480])
    ax.set_xticks([400, 480])
    ax.set_xticklabels([r'$400$', r'$4800$'], fontsize=11)
    ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
    ax.xaxis.get_majorticklabels()[1].set_horizontalalignment('right')
    ax.set_xlabel('time [samples]', labelpad=-11)
    ax.set_yticks([])

    legend(loc=(0.55, 0.1), framealpha=0.95)

plot_the_plot(sep)




def plot_latent_space_for_paper(sep):

    fig = figure(figsize=(4,4))

    plot_latent_space_impl(
        fig, [0.1, 0.1, 0.95, 0.95], sep.frames, sep.encoder,
        method='hist2', max_num_frames=None, bins=70, pow=.6)

    fig.text(0.83, 0.87, r'$p(z_n)$', fontsize=16)


plot_latent_space_for_paper(sep)
