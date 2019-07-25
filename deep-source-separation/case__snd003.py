from separator import *

cel2_ofs = 45000
cel2_wav = W.loadwav('./sounds/195138__flcellogrl__cello-tuning.wav')
cel2 = LazyGenerator(lambda n: 2.*cel2_wav[cel2_ofs:cel2_ofs+n,0])
laa_ofs = 40000
laa_wav = W.loadwav('./sounds/39914__digifishmusic__katy-sings-laaoooaaa.wav')
laa = LazyGenerator(lambda n: laa_wav[laa_ofs:laa_ofs+n,0])

sig1, sig2 = cel2, laa
signal = sig1 + sig2
POWER = 8
FRAME_SIZE, STRIDE = 2 ** POWER, 2
N_FRAMES = 80000

sep = Separator(
    signal=signal(N_FRAMES),
    stride=STRIDE,
    coder_factory=ConvFactory(
        input_size=FRAME_SIZE,
        latent_sizes=[5, 5],
        kernel_size=3,
        features=[24] * POWER,
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        decoder_noise={"stddev": .1, "decay": 0.00001, "final_stddev": 0.02, "correlation_dims":'all'},
    ),
    #input_noise={"stddev": .5, "decay": 0.00001, "final_stddev": 0.01},
    latent_noise={"stddev": .1, "decay": 0.00001, "final_stddev": 0.02, "correlation_dims":[5, 5]},
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Nadam(lr=0.001),
    loss='mae',
)

mix = signal(N_FRAMES)

W.savewav("./sndpair_03__mix.wav", 0.5*mix, 44100)
W.savewav("./sndpair_03__src1.wav", sig1(N_FRAMES), 44100)
W.savewav("./sndpair_03__src2.wav", sig2(N_FRAMES), 44100)

train_and_summary(sep, 100, batch_size=16)

if True:
    m1, m2 = sep.modes[0].infer(N_FRAMES), sep.modes[1].infer(N_FRAMES)
    W.savewav("./sndpair_03__sep1.wav", m1, 44100)
    W.savewav("./sndpair_03__sep2.wav", m2, 44100)
