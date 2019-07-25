from separator import *


sound_pairs = [
    {
        'sounds': [
            './sounds/195138__flcellogrl__cello-tuning.wav',
            './sounds/39914__digifishmusic__katy-sings-laaoooaaa.wav'],
        'offset': [45000, 45000],
        'power': 8
    },
    {
        'sounds': [
            '../kevin/sounds/328727__hellska__flute-note-tremolo.wav',
            '../kevin/sounds/248355__mtg__clarinet-g-3__16bit44k.wav'],
        'offset': [40000, 40000],
        'power': 7
    },
    {
        'sounds': [
            '../kevin/sounds/145513__zabuhailo__singingglass-mono__16bit44k.wav',
            '../kevin/sounds/203742__v4cuum__basbow-a29.wav'],
        'offset': [0, 0],
        'power': 8
    },
    {
        'sounds': [
            './sounds/195138__flcellogrl__cello-tuning.wav',
            '../kevin/sounds/203742__v4cuum__basbow-a29.wav'],
        'offset': [45000, 45000],
        'power': 8
    },
]

SOUND_IDX = 1   # select sound here
sound_pair = sound_pairs[SOUND_IDX]

POWER = sound_pair['power']   # the different sounds might need different window sizes
FRAME_SIZE, STRIDE = 2 ** POWER, 2
LATENT_SPACE = [5, 5]
NUM_FEATURES = 24
N_FRAMES = 2 * 44100


ofs1 = sound_pair['offset'][0]
wav1 = W.loadwav(sound_pair['sounds'][0])
sig1 = LazyGenerator(lambda n: wav1[ofs1:ofs1 + n, 0])
ofs2 = sound_pair['offset'][1]
wav2 = W.loadwav(sound_pair['sounds'][1])
sig2 = LazyGenerator(lambda n: wav2[ofs2:ofs2+n,0])

signal = sig1 + sig2



def make_sep():
    return Separator(
        signal=signal(N_FRAMES),
        stride=STRIDE,
        coder_factory=ConvFactory(
            input_size=FRAME_SIZE,
            latent_sizes=LATENT_SPACE,
            kernel_size=3,
            features=[NUM_FEATURES] * POWER,
            upsample_with_zeros=True,
            activation=leaky_tanh(0),
        ),
        latent_noise={"stddev": .1, "decay": 0.00001, "final_stddev": 0.0, "correlation_dims": LATENT_SPACE},
        input_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.0},
        signal_gens=[sig1, sig2],
        optimizer=keras.optimizers.Nadam(lr=0.002),
        loss='mae',
    )

sep = make_sep()
mix = signal(N_FRAMES)

seps = []
for _ in range(10):
   sep = make_sep()
   seps.append(sep)
   #train_and_summary(sep, n_epochs=15, batch_size=8)
   #train_and_summary(sep, n_epochs=20, batch_size=16)

for n, sep in enumerate(seps):
    # --- storing ----------
    # sep.model.save_weights("./sndpair_idx=01_run{:02}_only_corr_lat_noise.hdf5".format(n))
    # np.savetxt("./sndpair_idx=01_run{:02}_mi_only_corr_lat_noise.txt".format(n), sep.sep_recorder.mutual_information)
    # --- loading ----------
    sep.model.load_weights("./models_sndpair/sndpair_idx=01_run{:02}.hdf5".format(n))
    sep.sep_recorder.mutual_information = np.loadtxt("./models_sndpair/sndpair_idx=01_run{:02}_mi.txt".format(n))
    pass


# figure()
# for sep in seps:
#     plot(sep.sep_recorder.mutual_information, 'k')
# 
# for sep in seps:
#     train_and_summary(sep, 0)
# 


def plot_for_paper(seps):

    def plot_sep(ax, sep):
        src1, src2 = sep.signal_gens
        mode1, mode2 = sep.modes
        ax.plot(sig1(1000)[500:1000], '--k')
        ax.plot(sig2(1000)[500:1000], '--k')
        ax.plot(mode1.infer(1000)[500:1000], '-', color='#4466bb')
        ax.plot(mode2.infer(1000)[500:1000], '-', color='#44bb66')
        ax.set_xlim([0, 500])
        #ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    fig = figure(figsize=(8,4))

    ax = fig.add_axes([0.1, 0.15, 0.5, 0.8])

    for n, sep in enumerate(seps):
        lw = 3 if n in [3,4,6] else 1
        ax.semilogy(sep.sep_recorder.mutual_information + 0.0029, '-k', linewidth=lw)

    ax.set_xlabel("epoch", fontsize=16)
    ax.set_ylabel("mutual information", fontsize=16)
    ax.set_xticks([1, 5, 10, 15])

    ylim([0, 5])

    sep = seps[3]
    ax = fig.add_axes([0.62, 0.15, 0.35, 0.2])
    plot_sep(ax, sep)
    xlabel("time [samples]", fontsize=14)

    sep = seps[4]
    ax = fig.add_axes([0.62, 0.75, 0.35, 0.2])
    plot_sep(ax, sep)

    sep = seps[6]
    ax = fig.add_axes([0.62, 0.45, 0.35, 0.2])
    plot_sep(ax, sep)

    fig_ax = fig.add_axes([0, 0, 1, 1])
    fig_ax.patch.set_alpha(0)
    fig_ax.axis('off')

    fig_ax.arrow(0.65, 0.32, -0.052, 0.06, head_width=0.01, head_length=0.02, fc='k', ec='k')
    fig_ax.arrow(0.65, 0.62, -0.052, 0.06, head_width=0.01, head_length=0.02, fc='k', ec='k') 
    fig_ax.arrow(0.64, 0.775, -0.037, 0.0, head_width=0.01, head_length=0.02, fc='k', ec='k')  
    
    return fig_ax



def plot_for_paper2(seps):

    def plot_sep(ax, sep):
        src1, src2 = sep.signal_gens
        mode1, mode2 = sep.modes
        ax.plot(sig1(1000)[500:1000], '--k')
        ax.plot(sig2(1000)[500:1000], '--k')
        ax.plot(mode1.infer(1000)[500:1000], '-', color='#4466bb')
        ax.plot(mode2.infer(1000)[500:1000], '-', color='#44bb66')
        ax.set_xlim([0, 500])
        #ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    fig = figure(figsize=(8,3))

    ax = fig.add_axes([0.1, 0.15, 0.5, 0.8])

    epochs = np.arange(0, len(seps[0].sep_recorder.mutual_information)) + 1
    for n, sep in enumerate(seps):
        lw = 3 if n in [3,4,6] else 1
        ax.semilogy(epochs, sep.sep_recorder.mutual_information + 0.0029, '-k', linewidth=lw)

    ax.set_xlabel("epoch", fontsize=14, labelpad=-1)
    ax.set_ylabel("mutual information", fontsize=14)
    ax.set_xticks([1, 5, 10, 15])

    #ylim([0, 5])

    sep = seps[3]
    ax = fig.add_axes([0.62, 0.15, 0.35, 0.25])
    plot_sep(ax, sep)
    xlabel("time [samples]", fontsize=14, labelpad=-1)

    sep = seps[6]
    ax = fig.add_axes([0.62, 0.425, 0.35, 0.25])
    plot_sep(ax, sep)
    ax.xaxis.set_ticks([])

    sep = seps[4]
    ax = fig.add_axes([0.62, 0.7, 0.35, 0.25])
    plot_sep(ax, sep)
    ax.xaxis.set_ticks([])

    fig_ax = fig.add_axes([0, 0, 1, 1])
    fig_ax.patch.set_alpha(0)
    fig_ax.axis('off')
    fig_ax.set_xlim([0,1])
    fig_ax.set_ylim([0,1])

    fig_ax.arrow(0.65, 0.34, -0.052, 0.06, head_width=0.01, head_length=0.02, fc='k', ec='k')
    fig_ax.arrow(0.65, 0.64, -0.052, 0.06, head_width=0.01, head_length=0.02, fc='k', ec='k') 
    fig_ax.arrow(0.63, 0.9, -0.037, -0.07, head_width=0.01, head_length=0.02, fc='k', ec='k')  
    
    return fig_ax



def plot_latent_space_for_paper(sep):

    fig = figure(figsize=(4,4))

    plot_latent_space_impl(
        fig, [0.1, 0.1, 0.95, 0.95], sep.frames, sep.encoder,
        method='hist2', max_num_frames=60000, bins=80, pow=.9)

    fig.text(0.87, 0.90, r'$p(z_n)$', fontsize=15)

