# TODO
# • stability vs win size
# • try out using 'smoothed' past to predict
# • damped
# • multiple frequencies
# • amplitude modulated


from imports import *
from keras_tools import extra_layers as XL
from keras_tools import functional_layers as F
from pylab import imshow

frame_size = 128
shift = 8
n_latent = 4
noise_stddev = 0#.1   # if noise is strong almost always damped, if small it can still explode
noise_level = 0
n_pairs = 1000

use_bias = True
n_epochs = 100 # 300


n_latent = 2
make_signal = lambda n: np.sin(0.05*np.arange(n))
#n_latent = 3  # will not be able to learn both frequencies without non-linearities to fold torus into 3D
#n_latent = 4
#make_signal = lambda n: np.sin(0.05*np.arange(n)) + 0.3*np.sin(0.2212*np.arange(n))

#make_signal = lambda n: tools.add_noise(lorenz(n), 0.03)


# loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'
loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

def print_layer_outputs(model):
   for l in model.layers:
      print(l.output_shape[1:])


in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

dense = lambda s: F.dense(s, activation=None, use_bias=use_bias)

def make_dense_model(example_frame, latent_size):
   sig_len = np.size(example_frame)
   x = F.input_like(example_frame)
   xi = F.noise(noise_stddev)

   encoder = dense([latent_size])
   decoder = dense([sig_len])
   y = xi >> encoder >> decoder
   latent = encoder(x)
   out = decoder(latent)
   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [latent]), XL.jacobian(latent,x)



model, model2, encoder, dhdx = make_dense_model(in_frames[0], n_latent)

loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred) + 0.01*K.sum(dhdx*dhdx)

# 2D conv-model
#model, model2, encoder = make_model_2d(in_frames[0], n_latent)

print('model shape')
print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()


#model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#tools.train(model, tools.add_noise(in_frames, noise_stddev), out_frames[0], 20, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
tools.train(model, tools.add_noise(in_frames, noise_stddev), out_frames[0], 20, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
tools.train(model, tools.add_noise(in_frames, noise_stddev), out_frames[0], 50, n_epochs, loss_recorder)
#
#
#model2.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
#tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 32, 2*n_epochs, loss_recorder)
#
#model2.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
#tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 32, n_epochs, loss_recorder)
#
#model2.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
#tools.train(model2, tools.add_noise(in_frames, noise_level), out_frames, 64, n_epochs, loss_recorder)
#
#model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)


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
   frame_ = frame.reshape([1] + list(in_frames[0].shape))
   frames = np.array([f[0] for f in generate_n_frames_from(frame_, int(n_samples/shift))])
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

figure()
plot(model.get_weights()[0], 'k')
plot(model.get_weights()[2].T, 'b')
