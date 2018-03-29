from imports import *
from predictors import *
#from test_signals import *
import pylab as pl
from pylab import *

frame_size = 128
# with strong downsampling
# shift = 4
# n_latent = 12  # 8 works well with DAE on fm-signal
shift = 8
n_latent = 4  # 8 works well with DAE on fm-signal
kern_len = 5
noise_stddev = 0.03
noise_level = 0.0#1
# n_pairs = 10000
n_pairs = 3000

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = True
n_epochs = 100 # 300

# loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'
loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)
   # keras.losses.mean_absolute_error(y_true, y_pred) \
   # + 0.1*keras.losses.mean_absolute_error(XL.power_spectrum(y_true), XL.power_spectrum(y_pred))
   # + keras.losses.mean_absolute_error(K.log(XL.power_spectrum(y_true)), K.log(XL.power_spectrum(y_pred)))
   # TODO lo/hipass: K.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)


n_nodes = 20
#n_latent = 16
#frame_size = 64
#n_latent = 24
#shift = 4

# customization wrapper for ginzburg-landau generator
def ginz_lan(n):
   x = ginzburg_landau(n_samples=n, n_nodes=n_nodes, beta=0.1+0.5j)
   #return abs(x[:,:,0] + 1j*x[:,:,1])
   return x[:,:,0]

#make_signal = lorenz
make_signal = lambda n: TS.lorenz(1*n, [1,0,0])[::1]
make_test_signal = lambda n: TS.lorenz(1*n, [0,1,0])[::1]
#make_signal = lambda n: TS.hindmarsh_rose4(10*n, [0,0,3,-10])[::10]
#make_signal = lambda n: tools.add_noise(lorenz(n), 0.03)

# two frequencies should live in 3d space
#make_signal = lambda n: np.sin(0.05*np.arange(n)) + 0.3*np.sin(0.2212*np.arange(n))
#n_latent = 3



in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)


activation = fun.bind(XL.tanhx, alpha=0.1)

act = lambda: L.Activation(activation)
dense = lambda s: F.dense(s, activation=None, use_bias=use_bias)
conv1d = lambda feat: F.conv1d(int(feat), kern_len, stride=1, activation=None, use_bias=use_bias)


def make_dense_model(example_frame, latent_size):
   sig_len = np.size(example_frame)
   x = F.input_like(example_frame)
   eta = F.noise(noise_stddev)

   assert latent_size <= sig_len/4

   enc1 = dense([int(sig_len/2)]) >> act()
   enc2 = dense([int(sig_len/4)]) >> act()
   enc3 = dense([latent_size]) >> act()
   dec3 = dense([int(sig_len/2)]) >> act()
   dec2 = dense([int(sig_len/4)]) >> act()
   dec1 = dense([sig_len]) >> act()

   encoder = enc1 >> enc2 >> enc3
   decoder = dec3 >> dec2 >> dec1
   y = eta >> encoder >> decoder
   latent = encoder(x)
   out = decoder(latent)
   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [latent]), XL.jacobian(latent,x)




def make_ar_model(sig_len):
   x = F.input_like(in_frames[0])
   eta = lambda: F.noise(noise_stddev)
   d1 = dense([int(sig_len/2)]) >> act()
   d2 = dense([int(sig_len/4)]) >> act()
   d3 = dense([int(sig_len/8)]) >> act()
   d4 = dense([1]) >> act()
   # y = eta() >> d1 >> d2 >> d3 >> d4
   # return M.Model([x], [y(x)])
   chain = d1 >> d2 >> d3 >> d4
   y1 = (eta() >> chain)(x)
   x2 = L.concatenate([XL.Slice(XL.SLICE_LIKE[:,1:])(x), y1])
   y2 = chain(x2)
   return M.Model([x], [y1]), M.Model([x], [y1, y2])


model, model2, encoder, dhdx = make_dense_model(in_frames[0], n_latent)
#loss_function = lambda y_true, y_pred: \
#   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred) + 0.02*K.sum(dhdx*dhdx)


def prediction_dist(num_pred=100, pred_frames=10):
   pred_len = frame_size * pred_frames
   sig = make_test_signal(num_pred*frame_size + pred_len)
   diffs = []
   for n in arange(num_pred):
      frame = sig[n*frame_size:(n+1)*frame_size].copy()
      sig_pred = predict_signal(model, frame, shift, pred_len)[:pred_len]
      diffs.append(sig_pred - sig[n*frame_size:n*frame_size+pred_len])
   return np.std(np.array(diffs), axis=0)



#ar_model, ar_model2 = make_ar_model(len(in_frames[0]))

print('model shape')
tools.print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()

# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')
# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')

# M.save_model(model, "timeshift_autoencoder_dense_64_32_10__shift8__just_fm-mode_.hdf5")
# model = M.load_model("timeshift_autoencoder_dense_64_32_10__shift8__just_fm-mode_.hdf5",
#                      custom_objects={'<lambda>': loss_function})


model2.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 32, 3*n_epochs, loss_recorder)


model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)


def train_ar_model():
   ar_model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
   tools.train(ar_model, in_frames, next_samples, 32, n_epochs, loss_recorder)

   ar_model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
   tools.train(ar_model, in_frames, next_samples, 32, n_epochs, loss_recorder)

   ar_model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
   tools.train(ar_model, in_frames, next_samples, 64, n_epochs, loss_recorder)


def train_ar_model2():
   ar_model2.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
   tools.train(ar_model2, in_frames, [next_samples, next_samples2], 32, n_epochs, loss_recorder)

   ar_model2.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
   tools.train(ar_model2, in_frames, [next_samples, next_samples2], 32, n_epochs, loss_recorder)

   ar_model2.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
   tools.train(ar_model2, in_frames, [next_samples, next_samples2], 64, n_epochs, loss_recorder)


def predict_ar_model(start_frame=in_frames[0], n_pred=100):
   result = start_frame
   frame = start_frame
   for _ in range(n_pred):
      result = np.concatenate([result, ar_model.predict(frame.reshape(1,-1))[0]])
      frame = result[-frame_size:]
   return result


figure()
semilogy(loss_recorder.losses)


def plot_orig_vs_reconst(n=0):
   fig = pl.figure()
   pl.plot(in_frames[n], 'k')
   pl.plot(out_frames[0][n], 'k')
   pl.plot(model.predict(in_frames[n:n+1])[0], 'r')

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((out_frames[0][::step] - model.predict(in_frames[::step])).T, 'k', alpha=0.2)

plot_orig_vs_reconst(0)


def plot_prediction(n=2000, signal_gen=make_signal):
   sig = signal_gen(n+100)
   pred_sig = predict_signal(model, sig[:frame_size], shift, n+100)
   fig, ax = pl.subplots(2,1)
   ax[0].plot(sig[:n], 'k')
   ax[0].plot(pred_sig[:n], 'r')
   ax[1].plot(sig[:n]-pred_sig[:n])


def rot(axis, theta):
   mat = np.eye(3,3)
   axis = axis/sqrt(np.dot(axis, axis))
   a = cos(theta/2.)
   b, c, d = -axis*sin(theta/2.)

   return np.array([
      [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
      [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
      [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
   ])


#plot_prediction_im(3000, make_signal)
plot_prediction(3000, make_signal)

#TS.plot3d(*dot(rot([1,0,1],-1.4),code.T), '-k', linewidth=0.5)

# sig = make_signal(60000)
# pred_sig = predict_signal(60000)
# import sounddevice as sd
# sd.play(sig, 44100)
# sd.play(pred_sig, 44100)
