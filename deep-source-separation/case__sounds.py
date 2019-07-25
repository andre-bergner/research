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

SOUND_IDX = 0   # select sound here
sound_pair = sound_pairs[SOUND_IDX]

POWER = sound_pair['power']   # the different sounds might need different window sizes
FRAME_SIZE, STRIDE = 2 ** POWER, 2
LATENT_SPACE = [5, 5]
NUM_FEATURES = 24
# LATENT_SPACE = [7, 8]
# NUM_FEATURES = 36
N_FRAMES = 2 * 44100

# # testing mix3 with asymmetric latent space
# LATENT_SPACE = [6, 3]
# POWER = 7


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
            #decoder_noise={"stddev": .2, "decay": 0.00001, "final_stddev": 0.02, "correlation_dims":'all'},
            decoder_noise={"stddev": .2, "decay": 0.00001, "final_stddev": 0.},
        ),
        #latent_noise={"stddev": .1, "decay": 0.00001, "final_stddev": 0.02, "correlation_dims": LATENT_SPACE},
        input_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.},
        signal_gens=[sig1, sig2],
        optimizer=keras.optimizers.Adam(lr=0.001),
        #optimizer=keras.optimizers.Nadam(lr=.001, beta_1=0.99, beta_2=0.999, schedule_decay=0),
        loss='mae',
    )

sep = make_sep()

mix = signal(N_FRAMES)

W.savewav("./sndpair_{:02}__mix.wav".format(SOUND_IDX), 0.5*mix, 44100)
W.savewav("./sndpair_{:02}__src1.wav".format(SOUND_IDX), sig1(N_FRAMES), 44100)
W.savewav("./sndpair_{:02}__src2.wav".format(SOUND_IDX), sig2(N_FRAMES), 44100)

#train_and_summary(sep, n_epochs=10, batch_size=8)
#train_and_summary(sep, n_epochs=25, batch_size=8)
#train_and_summary(sep, n_epochs=25, batch_size=16)
#train_and_summary(sep, n_epochs=25, batch_size=16)
#sep.model.save_weights('___.hdf5')
#train_and_summary(sep, n_epochs=50, batch_size=8)

#sep.train(n_epochs=200, batch_size=16)

if False:
    m1, m2 = sep.modes[0].infer(N_FRAMES), sep.modes[1].infer(N_FRAMES)
    W.savewav("./sndpair_{:02}__sep1.wav".format(SOUND_IDX), m1, 44100)
    W.savewav("./sndpair_{:02}__sep2.wav".format(SOUND_IDX), m2, 44100)


seps = []
for _ in range(10):
   sep = make_sep()
   seps.append(sep)
   train_and_summary(sep, n_epochs=15, batch_size=16)

figure()
for sep in seps: semilogy(sep.sep_recorder.mutual_information)