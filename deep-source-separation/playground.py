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
power = 8
frame_size, stride = 2 ** power, 2
n_frames = 40000

sep = Separator(
    signal=signal(n_frames),
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[6, 7],        # DIFF 6,8
        kernel_size=5,
        features=[32] * power,      # DIFF 20
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        #use_batch_norm=True,   # TODO
        decoder_noise={"stddev": .5, "decay": 0.00001, "final_stddev": 0.05, "correlation_dims":'all'},
        #decoder_noise=dict(stddev=0.8, decay=0.0001, final_stddev=0.05),
        #decoder_noise={'dropout': 0.3},
        #one_one_conv=True,
        #resnet=True,
        #skip_connection=True,
    ),
    input_noise={"stddev": .5, "decay": 0.00001, "final_stddev": 0.01},
    latent_noise={"stddev": .2, "decay": 0.00001, "final_stddev": 0.01, "correlation_dims":[6, 7]},
    #latent_noise={"stddev": .2, "decay": 0.00001, "final_stddev": 0.05, "correlation_dims":[5, 5]},
    signal_gens=[sig1, sig2],
    #optimizer=keras.optimizers.Adam(lr=0.0001),  worx
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss='mae',
    # custom_features=[ [20]*power, [24]*power ],
    # custom_noises=[
    #     lambda: F.noise(stddev=0.2, decay=0.0001, final_stddev=0.05, correlation_dims='all'),
    #     lambda: F.noise(stddev=0.5, decay=0.0001, final_stddev=0.05, correlation_dims='all')
    # ],
    #info_loss=0.5
    #optimizer=keras.optimizers.Adam(lr=0.01),
    #optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001),
    #vanishing_xprojection=True
    #critic_optimizer=keras.optimizers.Adam(0.0005),
    #adversarial=0.1,
    #critic_runs=10,
)

#sep.model.summary()
#train_and_summary(sep, 20, batch_size=32)
train_and_summary(sep, 20, batch_size=64)

mix = signal(n_frames)
# m1 = sep.modes[0].infer(40000)
# m2 = sep.modes[1].infer(40000)

W.savewav("./mix.wav", 0.5*mix, 44100)
# W.savewav("./src1.wav", sig1(n_frames), 44100)
# W.savewav("./src2.wav", sig2(n_frames), 44100)
# W.savewav("./sep1.wav", m1, 44100)
# W.savewav("./sep2.wav", m2, 44100)


