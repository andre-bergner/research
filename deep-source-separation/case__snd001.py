from separator import *

cello_dis3_wav = W.loadwav("./wav/cello5/4_D#3.wav")
cello_dis3 = LazyGenerator(lambda n: cello_dis3_wav[:n,0])
choir_e4_wav = W.loadwav("./wav/boychoir e4.wav")
choir_e4 = LazyGenerator(lambda n: choir_e4_wav[:n,0])

sig1, sig2 = choir_e4, cello_dis3
signal = sig1 + sig2
power = 8
frame_size, stride = 2 ** power, 1

sep = Separator(
    signal=signal(30000),
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[5, 5],
        kernel_size=5,
        features=[32] * power,
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        decoder_noise={"stddev": .1, "decay": 0.0001, "final_stddev": 0.05, "correlation_dims":'all'},  # TODO
    ),
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.0002),
    loss='mae',
)

sep.model.summary()
train_and_summary(sep, 20, batch_size=16)


n_frames = 40000
mix = signal(n_frames)
m1, m2 = sep.modes[0].infer(n_frames), sep.modes[1].infer(n_frames)

# W.savewav("./snd001_mix.wav", 0.5*mix, 44100)
# W.savewav("./snd001_src1.wav", sig1(n_frames), 44100)
# W.savewav("./snd001_src2.wav", sig2(n_frames), 44100)
# W.savewav("./snd001_sep1.wav", m1, 44100)
# W.savewav("./snd001_sep2.wav", m2, 44100)


