from separator import *

sin1 = make_sin_gen(1.)
sin2 = make_sin_gen(1./np.pi)
sig1, sig2 = sin1, sin2

signal = sig1 + sig2

N_FRAMES = 10000
POWER = 6
STRIDE = 1
FRAME_SIZE = 2 ** POWER
LATENT_SPACE = [2, 2]

coder_factory=ConvFactory(
    input_size=FRAME_SIZE,
    latent_sizes=LATENT_SPACE,
    kernel_size=3,
    features=[16] * POWER,
    upsample_with_zeros=True,
    activation=leaky_tanh(0),
    decoder_noise={"stddev": 0.5, "decay": 0.001, "final_stddev": 0.},
)

def make_sep():
    return Separator(
        signal=signal(N_FRAMES),
        stride=STRIDE,
        coder_factory=coder_factory,
        signal_gens=[sig1, sig2],
        loss='mae',
        #input_noise={"stddev": 1.0, "decay": 0.0001, "final_stddev": 0.},
        optimizer=keras.optimizers.Adam(0.001),
    )

sep = make_sep()
train_and_summary(sep, n_epochs=10, batch_size=16)


# seps = []
# for n in range(10):
#     sep = make_sep()
#     seps.append(sep)
#     print("run {}/10".format(n+1))
#     sep.train(10, batch_size=8)
#     sep.train(10, batch_size=16)
#     training_summary(sep)


def plot_latent_space_for_paper(sep):

    fig = figure(figsize=(3, 3))

    plot_latent_space_impl(
        fig, [0.1, 0.1, 0.95, 0.95], sep.frames, sep.encoder,
        method='hist2', max_num_frames=None, bins=80, pow=1)

    fig.text(0.80, 0.84, r'$p(z_n)$', fontsize=16)


plot_latent_space_for_paper(sep)
