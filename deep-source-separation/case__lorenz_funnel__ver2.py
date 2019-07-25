print("""
    --------------------------------------------------------
    LORENZ + FUNNEL (RÖSSLER)
    
    This example  demonstrates a successful  separation of a
    Lorenz  time series  and a Rössler oscillator  in the so
    called funnel regime.
    --------------------------------------------------------
""")

from separator import *

sig1, sig2 = 0.5*lorenz, 0.5*funnel
#sig1, sig2 = 0.5*lorenz, 0.2*funnel
sig1, sig2 = 0.25*lorenz, 0.5*funnel
signal = sig1 + sig2
power = 8
frame_size = 2 ** power
space = [3, 4]
#space = [3, 5]

sep = Separator(
    signal=signal(10000),
    stride=2,
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=space,
        features=[16] * power,      # TODO add possibility to specify independently
        kernel_size=9,
        #kernel_size=3,
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        #activation=strong_leaky_tanh,
        #skip_connection=True,
        decoder_noise={"stddev": .2, "decay": 0.00001, "final_stddev": 0.05, "correlation_dims":'all'},
        #decoder_noise={"dropout": 0.01},
        #decoder_kernel_regularizer=keras.regularizers.l2(0.00001),
    ),
    loss='mae',
    #loss='logcosh',
    input_noise={"stddev": .2, "decay": 0.0001},
    #input_noise={"stddev": 1, "decay": 0.00001},
    latent_noise={
        "stddev": .3, "decay": 0.0001, "final_stddev": 0.02,
        "correlation_dims": space
    },
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.0002),
    #info_loss=0.05,
    #vanishing_xprojection=True,
    #custom_features=[ [16]*7, [24]*7 ],
    #custom_noises=[
    #    lambda: F.noise(stddev=0.1, decay=0.0001, final_stddev=0.05, correlation_dims='all'),
    #    lambda: F.noise(stddev=0.4, decay=0.0001, final_stddev=0.05, correlation_dims='all')
    #],
)


sep.model.summary()
train_and_summary(sep, n_epochs=50, batch_size=16)


seps = []
for _ in range(0):
#for _ in range(10):
    sep = Separator(
        signal=signal(10000),
        stride=2,
        coder_factory=ConvFactory(
            input_size=frame_size,
            latent_sizes=[4, 4],
            features=[16] * power,
            kernel_size=5,
            upsample_with_zeros=True,
            activation=leaky_tanh(0),
            decoder_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.05},
        ),
        loss='mae',
        latent_noise={
            "stddev": .3, "decay": 0.0001, "final_stddev": 0.02,
            "correlation_dims": [4, 4]
        },
        signal_gens=[sig1, sig2],
        optimizer=keras.optimizers.Adam(lr=0.0002),
    )
    seps.append(sep)
    train_and_summary(sep, n_epochs=30, batch_size=16)


# stride = 2
# frames1 = np.array([w[::stride] for w in windowed(sig1(10000), stride*frame_size, 1)])
# frames2 = np.array([w[::stride] for w in windowed(sig2(10000), stride*frame_size, 1)])
# cheater = M.Model([sep.modes[0].input], [sep.modes[0].output, sep.modes[1].output])
# cheater.compile(loss='mae', optimizer=keras.optimizers.Adam(0.001))
# lr = tools.train(cheater, sep.frames, [frames1, frames2], 32, 30)
# train_and_summary(sep, 0)