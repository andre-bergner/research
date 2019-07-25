from separator import *

cello_dis3_wav = W.loadwav("./wav/cello5/4_D#3.wav")
cello_dis3 = LazyGenerator(lambda n: cello_dis3_wav[:n,0])
choir_e4_wav = W.loadwav("./wav/boychoir e4.wav")
choir_e4 = LazyGenerator(lambda n: choir_e4_wav[:n,0])

cel2_wav = W.loadwav('./sounds/195138__flcellogrl__cello-tuning.wav')
cel2 = LazyGenerator(lambda n: 2.*cel2_wav[45000:45000+n,0])
ten_wav = W.loadwav('./sounds/82986__tim-kahn__countdown.wav')
ten = LazyGenerator(lambda n: 0.5*ten_wav[4000:4000+n,0])
laa_wav = W.loadwav('./sounds/39914__digifishmusic__katy-sings-laaoooaaa.wav')
laa = LazyGenerator(lambda n: laa_wav[40000:40000+n,0])

#sig1, sig2 = 0.6*lorenz, 0.4*fm_strong0
#sig1, sig2 = choir_e4, cello_dis3
#sig1, sig2 = cel2, ten
sig1, sig2 = cel2, laa
signal = sig1 + sig2
power = 7
frame_size, stride = 2 ** power, 2
n_frames = 40000

sep = Separator(
    signal=signal(n_frames),
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[8, 8],
        kernel_size=3,
        features=[24] * power,
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        resnet=True,
    ),
    latent_noise={"stddev": .1, "decay": 0.00001, "final_stddev": 0.01, "correlation_dims":[8, 8]},
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.0001),
    loss='mae',
)

sep.model.summary()
train_and_summary(sep, 20, batch_size=16)

mix = signal(n_frames)
frames = np.array([w[::stride] for w in windowed(mix, stride*frame_size, 1)])
# m1 = build_prediction(sep.modes[0], frames, stride=stride)
# m2 = build_prediction(sep.modes[1], frames, stride=stride)

W.savewav("./mix.wav", 0.5*mix, 44100)
# W.savewav("./src1.wav", sig1(n_frames), 44100)
# W.savewav("./src2.wav", sig2(n_frames), 44100)
# W.savewav("./sep1.wav", m1, 44100)
# W.savewav("./sep2.wav", m2, 44100)


