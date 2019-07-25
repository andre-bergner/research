from separator import *

sig1, sig2 = 0.3*lorenz, 0.2*fm_strong#0
signal = sig1 + sig2
frame_size = 128

sep = Separator(
   signal=signal(20000),
   coder_factory=ConvFactory(
      input_size=frame_size,
      latent_sizes=[4, 4],
      # kernel_size=5,
      # features=[4, 8, 8, 16, 16, 16, 32],
      # dec_features=[16, 16, 8, 8, 4, 4, 2],
      kernel_size=5,
      features=[4, 4, 4, 8, 8, 8, 16],
      dec_features=[8, 4, 4, 4, 2, 2, 2],
      decoder_noise=dict(stddev=0.3, decay=0.001, final_stddev=0.05),
      #resnet=True,
   ),
   signal_gens=[sig1, sig2],
   optimizer=keras.optimizers.Adam(lr=0.002),
)

weights0 = sep.model.get_weights()

sep.model.summary()
print('-------------------------------------------------------')
print('SEPARATOR')
print('-------------------------------------------------------')
train_and_summary(sep, 20)

def plot_modes3(sep, n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(sep.modes[0], sep.frames, n), 'r')
   plot(build_prediction(sep.modes[1], sep.frames, n), 'r')


print('-------------------------------------------------------')
print('CHEATER')
print('-------------------------------------------------------')


weights1 = sep.model.get_weights()


frames1 = np.array([w for w in windowed(sig1(20000), frame_size, 1)])
frames2 = np.array([w for w in windowed(sig2(20000), frame_size, 1)])
cheater = M.Model([sep.modes[0].input], [sep.modes[0].output, sep.modes[1].output])
cheater.compile(loss='mse', optimizer=keras.optimizers.Adam(0.01))

sep.model.set_weights(weights0)
tools.train(cheater, [sep.frames], [frames1, frames2], 128, 20)
weights2 = sep.model.get_weights()

loss = lambda: (np.square(sep.frames[:2000] - sep.model.predict(sep.frames[:2000]))).mean()

def loss_landscape(explore=0):

   rnd_weights = [explore * np.random.randn(*w.shape) for w in weights1]
   losses = []
   for a in linspace(-.2,1.2,100):
      print( '\r', end='' )
      print( '{} -> 1.2'.format(a), end='')
      print( '', end='', flush=True )
      sep.model.set_weights([a*w1+r + (1-a)*w2+r for w1,w2,r in zip(weights1, weights2, rnd_weights) ])
      losses.append(loss())

   return losses

sep.model.set_weights(weights1)
plot_modes3(sep)
sep.model.set_weights(weights2)
plot_modes3(sep)

l = loss_landscape(0.)
figure()
plot(l)
