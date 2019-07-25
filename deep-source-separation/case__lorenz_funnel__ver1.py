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
signal = sig1 + sig2
power = 7
frame_size = 2 ** power
space = [4, 4]

sep = Separator(
    signal=signal(10000),
    stride=2,
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=space,
        features=[16] * power,
        kernel_size=5,
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        decoder_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.05},
    ),
    loss='mae',
    latent_noise={
        "stddev": .3, "decay": 0.0001, "final_stddev": 0.02,
        "correlation_dims": space
    },
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.0002),
)

sep.model.summary()
train_and_summary(sep, n_epochs=25, batch_size=16)


# seps = []
# for _ in range(10):
#     sep = Separator(
#         signal=signal(10000),
#         stride=2,
#         coder_factory=ConvFactory(
#             input_size=frame_size,
#             latent_sizes=[4, 4],
#             features=[16] * power,
#             kernel_size=5,
#             upsample_with_zeros=True,
#             activation=leaky_tanh(0),
#             decoder_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.05},
#         ),
#         loss='mae',
#         latent_noise={
#             "stddev": .3, "decay": 0.0001, "final_stddev": 0.02,
#             "correlation_dims": [4, 4]
#         },
#         signal_gens=[sig1, sig2],
#         optimizer=keras.optimizers.Adam(lr=0.0002),
#     )
#     seps.append(sep)
#     train_and_summary(sep, n_epochs=30, batch_size=16)
