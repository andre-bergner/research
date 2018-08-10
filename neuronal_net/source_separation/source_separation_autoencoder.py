# TODO
# • separating two lorenz with current model is hard even using cheater
#   and unstable when training model afterwards
#   → try out more complex model

# TRY
# • minimize cross-channel predictivity, i.e. it should be impossible for the model to predict
#   the x2 out z1 and vise versa
#   • perhaps formulate in terms of ShAE, i.e. it's not possible to build a sub-ShAE across
#     the channels
#   • adversarial style training.
# • simplify (remove layers) from decoder
# • simplify parallel AE
# • multi-pass ShAE

# IDEAS:
# • constraint on separated channels ? e.g. less fluctuations
# • SAE with with separation in z-space


from imports import *
from coder_factories import *
from keras_tools import test_signals as TS

factor = 1
frame_size = factor*128
n_pairs = 20000
n_latent1 = 3
n_latent2 = 3
latent_sizes = [n_latent1, n_latent2]
n_epochs = 10
noise_stddev = 0.05



#sig1, sig2 = two_sin
#sig1, sig2 = kicks_sin1
#sig1, sig2 = lorenz_fm
#sig1, sig2 = fm_twins
#sig1, sig2 = tanhsin1, sin2
#sig1, sig2 = tanhsin1, sin4
#sig1, sig2 = cello, clarinet
#sig1, sig2 = cello_dis3, choir_e4
#sig1, sig2 = fm_soft3, 0.5*fm_soft3inv
sig1, sig2 = make_sin_gen(np.pi*0.05) + 0.4*make_sin_gen(3*0.05), 0.7*make_sin_gen(0.3432)

sig_gen = sig1 + sig2
sig_gen_s = lambda n: sig1(n) + sig2(n+100)[100:]

frames, *_ = TS.make_training_set(sig_gen, frame_size=frame_size, n_pairs=n_pairs)
frames1, *_ = TS.make_training_set(sig1, frame_size=frame_size, n_pairs=n_pairs)
frames2, *_ = TS.make_training_set(sig2, frame_size=frame_size, n_pairs=n_pairs)



#factory = DenseFactory
factory = ConvFactory
model, encoder, model_sf, [mode1, mode2] = make_factor_model(
   frames[0], factory(frames[0], latent_sizes, use_batch_norm=False, scale_factor=factor),
   noise_stddev=noise_stddev, shared_encoder=True
)
loss_function = lambda y_true, y_pred: keras.losses.mean_squared_error(y_true, y_pred) #+ 0.001*K.sum(dzdx*dzdx)

model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001), loss=loss_function)
#model_sf.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
model.summary()
plot_model(model, to_file='ssae.png', show_shapes=True)

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
