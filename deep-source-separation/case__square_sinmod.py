from separator import *

#sinmod = make_sin_gen(np.pi*0.1)
sinmod = LazyGenerator(lambda n: np.sin(np.pi*0.1*np.arange(n) + 20*np.sin(0.00599291*np.arange(n))))
sin2 = make_sin_gen(0.1)
#sinmod, sin2 = sin2, sinmod
square = LazyGenerator(lambda n: np.tanh(1000*sin2(n)))
sig1, sig2 = square, sinmod

signal = sig1 + sig2

N_FRAMES = 5000
POWER = 6
STRIDE = 1
FRAME_SIZE = 2 ** POWER
LATENT_SPACE = [2, 3]

coder_factory=ConvFactory(
    input_size=FRAME_SIZE,
    latent_sizes=LATENT_SPACE,
    kernel_size=3,
    #features=[4, 4, 4, 8, 8, 16],
    features=[16]*6,
    upsample_with_zeros=True,
    activation=leaky_tanh(0),
    decoder_noise={"stddev": .8, "decay": 0.0005, "final_stddev": 0.05, "correlation_dims":'all'},
    # the correlated noise significantly improves separation and convergence!
)

sep = Separator(
    signal=signal(N_FRAMES),
    coder_factory=coder_factory,
    signal_gens=[sig1, sig2],
    input_noise=dict(stddev=0.8, decay=0.00005, final_stddev=0.01),
    loss='mae',
    optimizer=keras.optimizers.Adam(0.001),
)




# -------------------------------------------------------------------------
# generate plot for paper

def plot_modes(separator):

    fig = plt.figure(figsize=(16, 2))

    m1, m2 = sep.modes[0].infer(N_FRAMES), sep.modes[1].infer(N_FRAMES)
    s1, s2 = sep.signal_gens[0](N_FRAMES), sep.signal_gens[1](N_FRAMES)

    ax = fig.add_axes([0.04, 0.13, 0.66, 0.8])
    plot(m1[:1000], '-', linewidth=1, color='#668888', label='separated')
    plot(m2[:1000], '-', linewidth=1, color='#668888')
    plot(s1[:1000], '--k', linewidth=0.5, label='original')
    plot(s2[:1000], '--k', linewidth=0.5)
    xlim([0, 1000])
    xlabel('time (samples)', labelpad=-10)

    ax = fig.add_axes([0.73, 0.13, 0.23, 0.8])
    plot(m1[:80], '-', linewidth=1, color='#668888', label='separated')
    plot(m2[:80], '-', linewidth=1, color='#668888')
    plot(s1[:80], '--k', linewidth=0.5, label='original')
    plot(s2[:80], '--k', linewidth=0.5)
    xlabel('time (samples)', labelpad=-10)
    xticks([0, 20, 60, 80])
    xlim([0, 80])
    yticks([])

    legend(loc=(0.8, 0.2), framealpha=1)


#PLOTTING = True
#
#if PLOTTING:
#    sep.model.load_weights('case__square_sinmod.hdf5')
#    plot_modes(sep)

#sep.model.summary()
train_and_summary(sep, 10, 4)
train_and_summary(sep, 10, 8)
train_and_summary(sep, 30, 16)
#train_and_summary(sep, 20, 16)
#train_and_summary(sep, 30, 32)

# sep.model.load_weights('case__square_sinmod.hdf5')




#f = h5py.File('case__square_sinmod.hdf5')
#list(f.keys()) 