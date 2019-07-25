from adversarial_ica import *

num_samples = 10000

raw_signals = np.array([gen(num_samples) for gen in [fm_strong0, lorenz]])

mix = np.array([
   [ 100 ,  0.5 ],
   [ .33 , -1.  ]
])

signals = mix.dot(raw_signals)

#ica = AdversarialICA(signals, discriminator_penalty=10, optimizer=keras.optimizers.Adam(lr=0.005))
ica = AdversarialICA(
   signals,
   instance_noise=.01,
   optimizer=keras.optimizers.Adam(lr=0.001),
   critic_optimizer=keras.optimizers.Adam(lr=0.0001)
)
ica.train(500, batch_size=128, discriminator_runs=10)

plot_summary(ica)
