# TODO / IDEAS
# ✔ concatenate with x-fade
# ✔ try different shifts
# ✔ try different frame_size's
# • second-order prediction --> train model and model^2 with frame[+1] and frame[+2], respectively
# • DAE
# • DAE in second order prediction?
# • contracting AE
# • use distance between prediction and original as stopping/quality metric
# • test with Rössler & Lorenz signal
# • "average" over several succesfully learned models --> common structure?
# • impact of latent dim on prediction: min size, stability of to big?
# • measure:
#   • impact of timestep
#   • impact of latent dim
#   • impact of noise
#   • impact of ...
# • get 'normal' (conv-)autoencoder for signals working
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
from pylab import imshow

frame_size = 128
shift = 8
n_latent = 10  # 8 works well with DAE
noise_level = 0.01

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = True
n_epochs = 200

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

def make_signal(n_samples=10000):
   t = np.arange(n_samples)
   sin = np.sin
   signal = 4. * sin(0.2*t + 6*sin(0.017*t))
   # signal = 4. * sin(0.2*t + 6*sin(0.017*t) + 4*sin(0.043*t))
   # signal = 4. * sin(0.2*t + 6*sin(0.02*t)) # +  5. * sin(t/3.7 + .3)

   #signal = 5. * sin(t/3.7 + .3) \
   #       + 3. * sin(t/1.3 + .1) \
   #       + 4. * sin(0.2*t + 6*sin(0.02*t))
          #+ 4. * sin(0.7*t + 14*sin(0.1*t))
          #+ 2. * sin(t/34.7 + .7)
   return signal / 20

def make_training_set(frame_size=30, n_pairs=1000, shift=1):
   sig = make_signal(n_pairs + frame_size + shift)
   ns = range(n_pairs)
   xs = [sig[n:n+frame_size] for n in ns]
   ys = [sig[n+shift:n+frame_size+shift] for n in ns]
   return np.array(xs), np.array(ys)

in_frames, out_frames  = make_training_set(frame_size=frame_size, n_pairs=10000, shift=shift)

act = lambda: L.Activation('tanh')
activation = fun.bind(XL.tanhx, alpha=0.1)
#act = lambda: L.LeakyReLU(alpha=0.5)

def dense(units, use_bias=True):
   return fun.ARGS >> L.Dense(units=int(units), activation=None, use_bias=use_bias) >> act()


def make_dense_model(sig_len, latent_size):
   x = input_like(in_frames[0])
   #y = dense(sig_len/2) >> dense(sig_len)
   #y = dense(sig_len/2) >> dense(latent_size) >> dense(sig_len/2) >> dense(sig_len)
   # y = dense(sig_len/2) >> dense(sig_len/4) >> dense(latent_size) >> dense(sig_len/4) >> dense(sig_len/2) >> dense(sig_len)

   enc1 = dense(sig_len/2)
   enc2 = dense(sig_len/4)
   enc3 = dense(latent_size)
   dec3 = dense(sig_len/2)
   dec2 = dense(sig_len/4)
   dec1 = dense(sig_len)
   # y = enc1 >> enc2 >> enc3 >> dec3 >> dec2 >> dec1
   # return M.Model([x], [y(x)])

   enc = enc3(enc1(x))
   y = dec1(dec3(enc))
   return M.Model([x], [y]), XL.jacobian(enc,x)



def train(model, inputs, target, batch_size, n_epochs, loss_recorder):
   if type(model.output) == list:
      target = len(model.output) * [target]
   tools.train(model, inputs, target, batch_size, n_epochs, loss_recorder)


#model  = make_dense_model(len(in_frames[0]), n_latent)
model, dhdx  = make_dense_model(len(in_frames[0]), n_latent)

loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + 0.001*K.sum(dhdx*dhdx)

# loss_function = lambda y_true, y_pred: \
#    keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)


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

model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
tools.train(model, tools.add_noise(in_frames, noise_level), out_frames, 20, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
tools.train(model, tools.add_noise(in_frames, noise_level), out_frames, 20, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
tools.train(model, tools.add_noise(in_frames, noise_level), out_frames, 50, 2*n_epochs, loss_recorder)

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
   pl.plot(out_frames[n], 'k')
   pl.plot(model.predict(in_frames[n:n+1])[0], 'r')

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((out_frames[::step] - model.predict(in_frames[::step])).T, 'k', alpha=0.2)

plot_orig_vs_reconst(0)



def xfade_append(xs, ys, n_split):
   num_left = len(ys) - n_split
   fade = np.linspace(0, 1, num_left)
   xs[-num_left:] *= (1-fade)
   xs[-num_left:] += ys[:num_left] * fade
   return concatenate([xs, ys[-n_split:]])

def predict_signal(n_samples):
   frames = np.array([f[0] for f in generate_n_frames_from(in_frames[0:1], int(n_samples/shift))])
   # pred_sig = concatenate([ f[-shift:] for f in frames[1:] ])
   pred_sig = in_frames[0]
   for f in frames[0:]:
      pred_sig = xfade_append(pred_sig, f, shift)
   return pred_sig

def plot_prediction(n=2000):
   sig = make_signal(n+100)
   pred_sig = predict_signal(n+100)
   fig, ax = pl.subplots(2,1)
   ax[0].plot(sig[:n], 'k')
   ax[0].plot(pred_sig[:n], 'r')
   ax[1].plot(sig[:n]-pred_sig[:n])

plot_prediction(3000)

# sig = make_signal(60000)
# pred_sig = predict_signal(60000)
# import sounddevice as sd
# sd.play(sig, 44100)
# sd.play(pred_sig, 44100)
