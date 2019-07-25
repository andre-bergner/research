print("""
    --------------------------------------------------------
    MODULATED SINUSOID + SINUSOID
    
    This example  demonstrates a successful  separation of a
    pure sinusoid and a second  frequency modulates sinusoid
    modulated around the same frequency as the first one.
    --------------------------------------------------------
""")

from separator import *

sig1, sig2 = 0.3*fm_soft3, 0.3*make_sin_gen(np.pi * 0.1)
signal = sig1 + sig2
frame_size = 128

sep = Separator(
    signal=signal(10000),
    stride=2,
    coder_factory=ConvFactory(
        input_size=frame_size,
        latent_sizes=[4, 4],
        features=[16] * 7,
        kernel_size=3,
        upsample_with_zeros=True,
        activation=leaky_tanh(0),
        skip_connection=True,
        decoder_noise={"stddev": 2., "decay": 0.0001, "final_stddev": 0.01, "correlation_dims":'all'},
    ),
    loss='mae',
    #latent_noise=0.2,
    #input_noise={"stddev": 2.0, "decay": 0.0001, "final_stddev": 0.01},
    #latent_noise={"stddev": 0.8, "decay": 0.0001, "final_stddev": 0.01},
    #info_loss=0.1,
    signal_gens=[sig1, sig2],
    optimizer=keras.optimizers.Adam(lr=0.0002),
   # optimizer=keras.optimizers.Adam(0.0001),
   # critic_optimizer=keras.optimizers.Adam(0.0005),
   # adversarial=0.2,
   # critic_runs=10,
    #custom_features=[ [8]*6, [16]*6 ],
    #custom_noises=[
    #    lambda: F.noise(stddev=1, decay=0.0001, final_stddev=0.05, correlation_dims='all'),
    #    lambda: F.noise(stddev=0.5, decay=0.0001, final_stddev=0.05, correlation_dims='all')
    #],
)

sep.model.summary()
train_and_summary(sep, n_epochs=50, batch_size=16)
#train_and_summary(sep, n_epochs=50, batch_size=32)



def compare_instantaneous_frequencies():

    from scipy.signal import hilbert

    x1 = sig1(10000)
    x2 = sig2(10000)
    y1 = build_prediction(sep.modes[0], sep.frames)
    y2 = build_prediction(sep.modes[1], sep.frames)

    hx1 = hilbert(x1)
    hx2 = hilbert(x2)
    hy1 = hilbert(y1)
    hy2 = hilbert(y2)

    inst_freq = lambda h: imag((h[2:]-h[:-2])/h[1:-1])

    fx1 = inst_freq(hx1)
    fx2 = inst_freq(hx2)
    fy1 = inst_freq(hy1)
    fy2 = inst_freq(hy2)

    figure(figsize=(10,3))
    plot(fy1[1:-100], label='y1')
    plot(fy2[1:-100], label='y2')
    plot(fx1[1:-100], color='#111144', linewidth=1, label='x1')
    plot(fx2[1:-100], color='#441111', linewidth=1, label='x2')
    legend()
    xlim([1000,5000])
    tight_layout()
