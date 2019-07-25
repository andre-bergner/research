from separator import *

frame_size = 128
n_pairs = 40000
n_latent1 = 4
n_latent2 = 4
latent_sizes = [n_latent1, n_latent2]
n_epochs = 10
noise_stddev = 2.0
noise_decay = 0.002

sig1, sig2 = 0.3*lorenz, 0.15*fm_strong
sig_gen = sig1 + sig2

frames, *_ = TS.make_training_set(sig_gen, frame_size=frame_size, n_pairs=n_pairs)

factory = ConvFactory
model, encoder, model_sf, [mode1, mode2] = make_factor_model(
   frames[0],
   ConvFactory(
      frame_size
      latent_sizes,
   ),
   noise_stddev=noise_stddev,
   noise_decay=noise_decay,
   shared_encoder=True
)
loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) #+ 0.001*K.sum(dzdx*dzdx)

model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001), loss=loss_function)
model.summary()

#x = F.input_like(frames[0])
#cheater = M.Model([x], [mode1(x), mode2(x)])
#cheater.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)


loss_recorder = LossRecorder(lambda n,m: pred_error([mode1,mode2][n], frames, [sig1,sig2][m], 2048))

tools.train(model, frames, frames, 128, 1*n_epochs, loss_recorder)
#tools.train(model_sf, [frames[:-1], frames[1:]], frames[:-1], 128, 1*n_epochs, loss_recorder)
#tools.train(cheater, frames, [frames1, frames2], 32, n_epochs, loss_recorder)


from pylab import *

def plot_modes3(n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(mode1, frames, n), 'r')
   plot(build_prediction(mode2, frames, n), 'r')

code = encoder.predict(frames)

training_summary(model, mode1, mode2, encoder, sig_gen, sig1, sig2, frames, loss_recorder)



from sklearn.decomposition import FastICA, PCA

def ica(x, n_components, max_iter=1000):
   ica_trafo = FastICA(n_components=n_components, max_iter=max_iter)
   return ica_trafo.fit_transform(x)

# model_, *_, modes = make_factor_model(
#    sig_gen(2000), DilatedFactory(sig_gen(2000), latent_sizes, use_batch_norm=False, scale_factor=factor),
#    noise_stddev=noise_stddev
# )
