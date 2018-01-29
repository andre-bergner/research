# TODO / IDEAS
# ✔ concatenate with x-fade
# ✔ try different shifts
# ✔ try different frame_size's
# ✔ DAE --> works
# ✔ second-order prediction --> train model and model^2 with frame[+1] and frame[+2], respectively
# • train two signals simultiously (in same AE) (two attractors)
# • DAE + L2 loss
# • plot code
# • plot distances
# • try generating ginzburg landau (use 1d-conv)
# • try generating images/texture
# • DAE in second order prediction?
# • contracting AE
# • impact of latent_space size for noisy system!
# • use distance between prediction and original as stopping/quality metric
# • test with Rössler & Lorenz signal
# • compare against plain nonlineae-AR-networks
# • "average" over several succesfully learned models --> common structure?
# • impact of latent dim on prediction: min size, stability of to big?
# • measure:
#   • impact of timestep
#   • impact of latent dim
#   • impact of noise
#   • impact of batch-size
# • connection between signal time-scale and step-size
# • batch normalization
# • get 'normal' (conv-)autoencoder for signals working
# • try learning noisy time series (long time)
# • try to add wavelet-ae outer ring
# • try loss functions: fft, separate lopass & hipass filters
# • try adding noise (denoising timeshift AE)
# • try to increase frame_size using deep-conv with downsampling
# • try strided prediction (Taken's theorem) --> random sampling?
# • hyper plane time shift AE: learns to predict several signals
#   --> i.e. add the classic AE feature of learning disjoint entities that lie on a common manifold
#   --> additional to the manifold of the flow learn neighboring manifolds in the space of dynamical systems

from gen_autoencoder import *
from keras_tools import extra_layers as XL
from test_signals import *
from pylab import imshow

frame_size = 128
# with strong downsampling
# shift = 4
# n_latent = 12  # 8 works well with DAE on fm-signal
shift = 8
n_latent = 4  # 8 works well with DAE on fm-signal
noise_level = 0.01
# n_pairs = 10000
n_pairs = 4000

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = True
n_epochs = 100 # 300

learning_rate = .1
# loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'
loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)
   # keras.losses.mean_absolute_error(y_true, y_pred) \
   # + 0.1*keras.losses.mean_absolute_error(XL.power_spectrum(y_true), XL.power_spectrum(y_pred))
   # + keras.losses.mean_absolute_error(K.log(XL.power_spectrum(y_true)), K.log(XL.power_spectrum(y_pred)))
   # TODO lo/hipass: K.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)



def print_layer_outputs(model):
   for l in model.layers:
      print(l.output_shape[1:])

# customization wrapper for ginzburg-landau generator
def ginz_lan(n):
   x = ginzburg_landau(n_samples=n, n_nodes=10, beta=0.1+0.5j)
   return abs(x[:,:,0] + 1j*x[:,:,1])
   #return x[:,:,0]

#make_signal = lorenz
#make_signal = lambda n: lorenz(5*n)[::5]
make_signal = lambda n: tools.add_noise(lorenz(n), 0.03)

# make_signal = lambda n: ginz_lan(n)[:,5]
# n_latent = 20
# frame_size = 200
# shift = 4

in_frames, out_frames, next_samples = make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
# in_frames2, out_frames2 = make_training_set(make_signal2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
# in_frames, out_frames = concat([in_frames, in_frames2], [out_frames, out_frames2])

activation = fun.bind(XL.tanhx, alpha=0.1)
#act = lambda: L.Activation(activation)
#act = lambda: L.Activation('tanh')
#act = lambda: L.LeakyReLU(alpha=0.5)

def dense(units, use_bias=True):
   #return fun.ARGS >> L.Dense(units=int(units), activation=activation, use_bias=use_bias) >> act()
   return fun.ARGS >> L.Dense(units=int(units), activation=activation, use_bias=use_bias)


def make_dense_model(sig_len, latent_size):
   sig_len = np.size(in_frames[0])
   x = input_like(in_frames[0])
   #y = dense(sig_len/2) >> dense(sig_len)
   #y = dense(sig_len/2) >> dense(latent_size) >> dense(sig_len/2) >> dense(sig_len)
   # y = dense(sig_len/2) >> dense(sig_len/4) >> dense(latent_size) >> dense(sig_len/4) >> dense(sig_len/2) >> dense(sig_len)

   assert latent_size <= sig_len/4

   enc1 = dense(sig_len/2)
   enc2 = dense(sig_len/4)
   enc3 = dense(latent_size)
   dec3 = dense(sig_len/2)
   dec2 = dense(sig_len/4)
   dec1 = dense(sig_len)
   ##y = enc1 >> enc2 >> enc3 >> dec3 >> dec2 >> dec1
   #out = y(x)
   #return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [encoder(x)])

   encoder = enc1 >> enc2 >> enc3
   decoder = dec3 >> dec2 >> dec1
   y = encoder >> decoder
   enc_x = encoder(x)
   out = decoder(enc_x)
   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [enc_x]), XL.jacobian(enc_x,x)

def make_ar_model(sig_len):
   x = input_like(in_frames[0])
   d1 = dense(sig_len/2)
   d2 = dense(sig_len/4)
   d3 = dense(sig_len/8)
   d4 = dense(1)
   y = d1 >> d2 >> d3 >> d4
   return M.Model([x], [y(x)])


def train(model, inputs, target, batch_size, n_epochs, loss_recorder):
   if type(model.output) == list:
      target = len(model.output) * [target]
   tools.train(model, inputs, target, batch_size, n_epochs, loss_recorder)


model, model2, encoder, dhdx = make_dense_model(len(in_frames[0]), n_latent)

#model, dhdx, encoder  = make_dense_model(len(in_frames[0]), n_latent)
#loss_function = lambda y_true, y_pred: \
#   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred) + 0.02*K.sum(dhdx*dhdx)

ar_model = make_ar_model(len(in_frames[0]))

print('model shape')
print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()

# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')
# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')

# M.save_model(model, "timeshift_autoencoder_dense_64_32_10__shift8__just_fm-mode_.hdf5")
# model = M.load_model("timeshift_autoencoder_dense_64_32_10__shift8__just_fm-mode_.hdf5",
#                      custom_objects={'<lambda>': loss_function})


# joint_model does only work with real auto encoders

#model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_level), out_frames[0], 20, n_epochs, loss_recorder)
#
#model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_level), out_frames[0], 20, n_epochs, loss_recorder)
#
#model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_level), out_frames[0], 50, n_epochs, loss_recorder)
#
#
model2.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 20, n_epochs, loss_recorder)

model2.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 20, n_epochs, loss_recorder)

model2.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 50, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)


def train_ar_model():
   ar_model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
   tools.train(ar_model, in_frames, next_samples, 20, n_epochs, loss_recorder)

   ar_model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
   tools.train(ar_model, in_frames, next_samples, 20, n_epochs, loss_recorder)

   ar_model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
   tools.train(ar_model, in_frames, next_samples, 50, n_epochs, loss_recorder)


def predict_ar_model(start_frame=in_frames[0], n_pred=100):
   result = start_frame
   frame = start_frame
   for _ in range(n_pred):
      result = np.concatenate([result, ar_model.predict(frame.reshape(1,-1))[0]])
      frame = result[-frame_size:]
   return result




from pylab import *

figure()
semilogy(loss_recorder.losses)



import pylab as pl

def predict_n_steps_from_frame(start_frame=in_frames[0:1], n_pred=1):
   frame = start_frame
   #f = np.random.rand(*f.shape)
   for _ in range(n_pred):
      frame = model.predict(frame)
   return frame[0]

def predict_n_steps(n_start=0, n_pred=1):
   return predict_n_steps_from_frame(in_frames[n_start:n_start+1], n_pred)


def generate_n_frames_from(frame, n_frames=10):
   for n in range(n_frames):
      frame = model.predict(frame)
      yield frame

#def predict_n_steps_(n_start=0, n_pred=1):
#   f = in_frames[n_start:n_start+1]
#   #f = np.random.rand(*f.shape)
#   for _ in range(n_pred):
#      f = model.predict(f)
#   return f[0]


def plot_orig_vs_reconst(n=0):
   fig = pl.figure()
   pl.plot(in_frames[n], 'k')
   pl.plot(out_frames[0][n], 'k')
   pl.plot(model.predict(in_frames[n:n+1])[0], 'r')

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((out_frames[0][::step] - model.predict(in_frames[::step])).T, 'k', alpha=0.2)

plot_orig_vs_reconst(0)



def xfade_append(xs, ys, n_split):
   num_left = len(ys) - n_split
   fade = np.linspace(0, 1, num_left)
   xs[-num_left:] *= (1-fade)
   xs[-num_left:] += ys[:num_left] * fade
   return concatenate([xs, ys[-n_split:]])

def predict_signal(n_samples, frame):
   frames = np.array([f[0] for f in generate_n_frames_from(frame.reshape(1,-1), int(n_samples/shift))])
   # pred_sig = concatenate([ f[-shift:] for f in frames[1:] ])
   pred_sig = frame
   for f in frames[0:]:
      pred_sig = xfade_append(pred_sig, f, shift)
   return pred_sig

def plot_prediction(n=2000, signal_gen=make_signal):
   sig = signal_gen(n+100)
   pred_sig = predict_signal(n+100, sig[:frame_size])
   fig, ax = pl.subplots(2,1)
   ax[0].plot(sig[:n], 'k')
   ax[0].plot(pred_sig[:n], 'r')
   ax[1].plot(sig[:n]-pred_sig[:n])

plot_prediction(3000, make_signal)
#plot_prediction(3000, make_signal2)

# sig = make_signal(60000)
# pred_sig = predict_signal(60000)
# import sounddevice as sd
# sd.play(sig, 44100)
# sd.play(pred_sig, 44100)
