from imports import *
from keras_tools import extra_layers as XL
from keras_tools import functional_layers as F
#from test_signals import *
from pylab import imshow

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
make_signal = lambda n: TS.lorenz(1*n)[::1]
#make_signal = lambda n: tools.add_noise(lorenz(n), 0.03)

# two frequencies should live in 3d space
#make_signal = lambda n: np.sin(0.05*np.arange(n)) + 0.3*np.sin(0.2212*np.arange(n))
#n_latent = 3

# make_signal = lambda n: ginz_lan(n)[:,5]
# n_latent = 20
# frame_size = 200
# shift = 4

# make_signal = lambda n: ginz_lan(n)


in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
# in_frames2, out_frames2 = make_training_set(make_signal2, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
# in_frames, out_frames = concat([in_frames, in_frames2], [out_frames, out_frames2])

# in_frames = in_frames.transpose(0,2,1)
# out_frames = [out_frames[0].transpose(0,2,1), out_frames[1].transpose(0,2,1)]


activation = fun.bind(XL.tanhx, alpha=0.1)
#act = lambda: L.Activation(activation)
#act = lambda: L.Activation('tanh')
#act = lambda: L.LeakyReLU(alpha=0.5)

# def dense(units, use_bias=True):
#    return fun.ARGS >> L.Dense(units=int(units), activation=activation, use_bias=use_bias)

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



def make_model_2d(example_frame, latent_size, simple=True):
   sig_len = example_frame.shape[-1]
   #x = F.noise(noise_stddev)(F.input_like(example_frame))
   x = F.input_like(example_frame)
   eta = F.noise(noise_stddev)

   # tp = fun._ >> L.Permute((1, 2))  TODO: try using permute/transpose

   if simple:

      print("simple model")

      enc1 = conv1d(sig_len/2) >> act() # >> F.dropout(0.2)
      enc2 = conv1d(sig_len/4) >> act() # >> F.dropout(0.2)
      #enc3 = conv1d(latent_size)
      enc3 = F.flatten() >> dense([n_latent]) >> act() # >> F.batch_norm()

      # TODO: figure out dimension from shape
      dec3 = dense([n_nodes, int(sig_len/4)]) >> act() # >> F.batch_norm() >> F.dropout(0.2)
      #dec3 = conv1d(sig_len/4)
      dec2 = conv1d(sig_len/2) >> act() # >> F.dropout(0.2)
      dec1 = conv1d(sig_len) >> act()

   else:

      enc1 = conv1d(sig_len/2) >> act() >> F.dropout(0.2)
      enc2a = conv1d(sig_len/4) >> act() >> F.dropout(0.2)
      enc2b = conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
      enc2 = enc2a >> enc2b
      enc3 = F.flatten() >> dense([n_latent]) >> act() >> F.batch_norm()
      enc3 = enc3 >> F.flatten() >> dense([n_latent]) >> act() >> F.batch_norm()

      # TODO: figure out dimension from shape
      dec3 = dense([n_nodes, int(sig_len/4)]) >> act() >> F.batch_norm() >> F.dropout(0.2)
      dec2b = conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
      dec2a = conv1d(sig_len/2) >> act() >> F.dropout(0.2)
      dec2 = dec2b >> dec2a
      dec1 = conv1d(sig_len) >> act()

      # dec4 = up(2) >> conv(8, 1) >> act()

   encoder = enc1 >> enc2 >> enc3
   decoder = dec3 >> dec2 >> dec1
   y = encoder >> decoder
   latent = encoder(x)
   out = decoder(latent)
   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [latent])#, XL.jacobian(latent,x)


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

# 2D conv-model
#model, model2, encoder = make_model_2d(in_frames[0], n_latent)


ar_model, ar_model2 = make_ar_model(len(in_frames[0]))

print('model shape')
tools.print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()

# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')
# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')

# M.save_model(model, "timeshift_autoencoder_dense_64_32_10__shift8__just_fm-mode_.hdf5")
# model = M.load_model("timeshift_autoencoder_dense_64_32_10__shift8__just_fm-mode_.hdf5",
#                      custom_objects={'<lambda>': loss_function})


# joint_model does only work with real auto encoders

#model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_stddev), out_frames[0], 20, n_epochs, loss_recorder)
#
#model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_stddev), out_frames[0], 20, n_epochs, loss_recorder)
#
#model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_stddev), out_frames[0], 50, n_epochs, loss_recorder)
#
#
model2.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 32, 2*n_epochs, loss_recorder)

model2.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 32, n_epochs, loss_recorder)

model2.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 64, n_epochs, loss_recorder)

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
   # expects xs,ys to have shape (N,...,1)
   # expects time in first dimension: N
   num_left = ys.shape[0] - n_split
   fade = np.linspace(0, 1, num_left)
   fade_tensor = np.tile(fade, list(ys.shape[1:]) + [1]).T
   xs[-num_left:] *= (1-fade_tensor)
   xs[-num_left:] += ys[:num_left] * fade_tensor
   return concatenate([xs, ys[-n_split:]], axis=0)

def predict_signal(n_samples, frame):
   frame_ = frame.reshape([1] + list(frame.shape))
   frames = np.array([f[0] for f in generate_n_frames_from(frame_, int(n_samples/shift))])
   # pred_sig = concatenate([ f[-shift:] for f in frames[1:] ])
   pred_sig = frame
   for f in frames[0:]:
      pred_sig = xfade_append(pred_sig.T, f.T, shift).T
   return pred_sig

def plot_prediction(n=2000, signal_gen=make_signal):
   sig = signal_gen(n+100)
   pred_sig = predict_signal(n+100, sig[:frame_size])
   fig, ax = pl.subplots(2,1)
   ax[0].plot(sig[:n], 'k')
   ax[0].plot(pred_sig[:n], 'r')
   ax[1].plot(sig[:n]-pred_sig[:n])

def plot_prediction_im(n=2000, signal_gen=make_signal):
   sig = signal_gen(n+100).T
   pred_sig = predict_signal2(n+100, sig[:,:frame_size])
   fig, ax = pl.subplots(3,1)
   ax[0].imshow(log(.1 + abs(sig[:n])), aspect='auto')
   ax[1].imshow(log(.1 + abs(pred_sig[:n])), aspect='auto')
   ax[2].imshow(log(.1 + abs(sig[:,:n]-pred_sig[:,:n])), aspect='auto')






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
