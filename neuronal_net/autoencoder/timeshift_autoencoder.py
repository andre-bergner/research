# TODO
# • concatenate with x-fade
# • try different shifts
# • try different frame_size's
# • try adding noise (denoising timeshift AE)
# • try to increase frame_size using deep-conv with downsampling
# • try strided prediction (Taken's theorem) --> random sampling?


from gen_autoencoder import *
from pylab import imshow

sig_len = 256
n_features = 4
kern_len = 16

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = False
n_epochs = 300

learning_rate = .1
loss_function = 'mean_absolute_error'

def model_gen(x, reshape_in, reshape_out, conv, act, up):

   enc1  = conv(2, kern_len, 1) >> act()
   enc1b = conv(2, 1, 2) >> act()
   enc2  = conv(4, kern_len, 1) >> act()
   enc2b = conv(4, 1, 2) >> act()
   enc3  = conv(8, kern_len, 1) >> act()
   enc3b = conv(8, 1, 2) >> act()
   enc4  = conv(16, kern_len, 1) >> act()
   enc4b = conv(16, 1, 2) >> act()

   #latent = conv(n_features, 2, 2, padding='valid') >> act()
   to_latent = conv(n_features, 1, 2) >> act()
   #to_latent = conv(16, 1, 2) >> act()
   from_latent = up(2) >> conv(16, 1) >> act()

   dec4  = up(2) >> conv(8, 1) >> act()
   dec4b =          conv(8, kern_len) >> act()
   dec3  = up(2) >> conv(4, 1) >> act()
   dec3b =          conv(4, kern_len) >> act()
   dec2  = up(2) >> conv(2, 1) >> act()
   dec2b =          conv(2, kern_len) >> act()
   dec1  = up(2) >> conv(1, 1) >> act()
   dec1b =          conv(1, kern_len) >> act()

   # ----- debug -----------------------------------
   # m = M.Model([x], [(reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc4 >> to_latent >> dec4 >> dec3 >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out)(x)])
   # for l in m.layers:
   #    print(l.output_shape[1:])

   y1 = reshape_in >> enc1 >> enc1b >> dec1 >> dec1b >> reshape_out
   y2 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y3 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y4 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> dec4 >> dec4b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y  = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> to_latent >> from_latent >> dec4 >> dec4b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out

   model = M.Model([x], [y(x)])
   joint_model = M.Model([x], [y1(x),y2(x),y3(x),y4(x),y(x)])

   return model, joint_model



def print_layer_outputs(model):
   for l in model.layers:
      print(l.output_shape[1:])

def make_signal(n_samples=10000):
   t = np.arange(n_samples)
   sin = np.sin
   signal = 5. * sin(t/3.7 + .3) \
          + 3. * sin(t/1.3 + .1) \
          + 4. * sin(0.2*t + 6*sin(0.02*t))
          #+ 4. * sin(0.7*t + 14*sin(0.1*t))
          #+ 2. * sin(t/34.7 + .7)
   return signal / 20

def make_training_set(frame_size=30, n_pairs=1000, shift=1):
   sig = make_signal(n_pairs + frame_size + shift)
   ns = range(n_pairs)
   xs = [sig[n:n+frame_size] for n in ns]
   ys = [sig[n+shift:n+frame_size+shift] for n in ns]
   return np.array(xs), np.array(ys)

in_frames, out_frames  = make_training_set(frame_size=64, n_pairs=10000, shift=10)  # 64

# model, joint_model = make_model([sig_len], model_gen, use_bias=use_bias)
# model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)
# joint_model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)

def dense(units, use_bias=True):
   return fun.ARGS >> L.Dense(units=int(units), activation='tanh', use_bias=use_bias)

sig_len = len(in_frames[0])

x = input_like(in_frames[0])
y = dense(sig_len/2) >> dense(10) >> dense(sig_len/2) >> dense(sig_len)

model = M.Model([x], [y(x)])
model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)





print('model shape')
print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()

# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')
# model.save_weights('timeshift_autoencoder_dense_64_32_10.hdf5')

model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
tools.train(model, in_frames, out_frames, 20, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
tools.train(model, in_frames, out_frames, 20, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
tools.train(model, in_frames, out_frames, 50, n_epochs, loss_recorder)

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


# sig = make_signal(60000)
# frames = np.array([f[0] for f in generate_n_frames_from(in_frames[0:1],6000)])
# pred_sig = concatenate([ f[-10:] for f in frames[1:] ])

# import sounddevice as sd
# sd.play(sig, 44100)
# sd.play(pred_sig, 44100)
