from imports import *
from predictors import *
#from test_signals import *
import pylab as pl
from pylab import *


# -----------------------------------------------------------------------------------
# MODEL PARAMETERS
# -----------------------------------------------------------------------------------

kern_len = 5
noise_stddev = 0.03
noise_level = 0.0#1
use_bias = True
n_nodes = 20
n_latent = 20
frame_size = 128
shift = 16
n_pairs = 1000
n_epochs = 100
noise_stddev = 0.01

#frame_size = 64
#n_nodes = 10
#n_latent = 40
#shift = 4

#n_pairs = 5000
#n_epochs = 20
#noise_stddev = 0.0

# try deep conv net with pooling, with resnet



loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)
   # keras.losses.mean_absolute_error(y_true, y_pred) \
   # + 0.1*keras.losses.mean_absolute_error(XL.power_spectrum(y_true), XL.power_spectrum(y_pred))
   # + keras.losses.mean_absolute_error(K.log(XL.power_spectrum(y_true)), K.log(XL.power_spectrum(y_pred)))

mse = keras.losses.mean_squared_error
mae = keras.losses.mean_absolute_error
diff1 = lambda x: x[:,2:] - x[:,:-2]
diff2 = lambda x: x[:,:,2:] - x[:,:,:-2]
#loss_function = lambda y_true, y_pred: mse(y_true, y_pred)[:,1:-1] + mse(diff1(y_true), diff1(y_pred))
#loss_function = lambda y_true, y_pred: 0.5*mae(y_true, y_pred) + mae(diff2(y_true), diff2(y_pred)) + mae(diff2(diff2(y_true)), diff2(diff2(y_pred)))
loss_function = lambda y_true, y_pred: mae(y_true, y_pred)


activation = fun.bind(XL.tanhx, alpha=0.1)
act = lambda: L.Activation(activation)
dense = lambda s: F.dense(s, activation=None, use_bias=use_bias)
conv1d = lambda feat: F.conv1d(int(feat), kern_len, stride=1, activation=None, use_bias=use_bias)


# -----------------------------------------------------------------------------------
# DATA GENERATION
# -----------------------------------------------------------------------------------

# customization wrapper for ginzburg-landau generator
# beta = 0.1 + 0.5j
# beta = 0.1 + 0.2j
beta = 0.2 + 0.2j
def ginz_lan(n):
   x = TS.ginzburg_landau(n_samples=n, n_nodes=n_nodes, beta=beta)
   return 0.4*abs(x[:,:,0] + 1j*x[:,:,1])
   #return x[:,:,0]

make_signal = lambda n: ginz_lan(n)#[:,5]

in_frames, out_frames, next_samples, _ = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
in_frames = in_frames.transpose(0,2,1)
out_frames = [out_frames[0].transpose(0,2,1), out_frames[1].transpose(0,2,1)]
next_samples = next_samples.reshape(-1,n_nodes,1)


def make_model_2d(example_frame, latent_size, simple=True):
   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

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
      dec1 = conv1d(sig_len) #>> act()

   else:

      enc1 = conv1d(sig_len/2) >> act() >> F.dropout(0.2)
      enc2a = conv1d(sig_len/4) >> act() >> F.dropout(0.2)
      enc2b = conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
      enc2 = enc2a >> enc2b
      enc3 = F.flatten() >> dense([n_latent]) >> act() >> F.batch_norm()

      # TODO: figure out dimension from shape
      dec3 = dense([n_nodes, int(sig_len/4)]) >> act() >> F.batch_norm() >> F.dropout(0.2)
      dec2b = conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
      dec2a = conv1d(sig_len/2) >> act() >> F.dropout(0.2)
      dec2 = dec2b >> dec2a
      dec1 = conv1d(sig_len) #>> act()

      # dec4 = up(2) >> conv(8, 1) >> act()

   encoder = eta() >> enc1 >> enc2 >> enc3
   decoder = eta() >> dec3 >> dec2 >> dec1
   y = encoder >> decoder
   latent = encoder(x)
   out = decoder(latent)
   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [latent])#, XL.jacobian(latent,x)



def make_model_2d_arnn(example_frame, simple=True):
   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   if simple:

      print("simple model")

      d1 = conv1d(sig_len/2) >> act()
      d2 = conv1d(sig_len/2) >> act()
      d3 = conv1d(sig_len/8) >> act()
      d4 = conv1d(sig_len/16) >> act()
      d5 = conv1d(1) >> act()

   else:

      pass

   y = d1 >> d2 >> d3 >> d4 >> d5
   return M.Model([x], [y(x)])


model, model2, encoder = make_model_2d(in_frames[0], n_latent, simple=False)


tools.print_layer_outputs(model)
loss_recorder = tools.LossRecorder()

model2.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
tools.train(model2, in_frames, out_frames, 32, n_epochs//20, loss_recorder)
model2.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
tools.train(model2, in_frames, out_frames, 128, n_epochs, loss_recorder)

#model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#tools.train(model, in_frames, out_frames[0], 32, n_epochs//20, loss_recorder)
#model.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
#tools.train(model, in_frames, out_frames[0], 128, n_epochs, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)


arnn_model = make_model_2d_arnn(in_frames[0], simple=True)
arnn_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_absolute_error)
arnn_loss_recorder = tools.LossRecorder()
tools.train(arnn_model, in_frames, next_samples, 32, n_epochs, arnn_loss_recorder)

figure()
semilogy(loss_recorder.losses)



def predict_ar_model(model, start_frame, n_samples):
   frame_size = start_frame.shape[-1]
   result = start_frame.copy()
   frame = start_frame.copy()
   frame_ = frame.reshape([1] + list(frame.shape))
   for _ in range(n_samples):
      result = np.concatenate([result, model.predict(frame_)[0]], axis=1)
      frame = result[:,-frame_size:]
      frame_ = frame.reshape([1] + list(frame.shape))
   return result






def plot_prediction_im(n=2000, signal_gen=make_signal, ofs=0):
   sig = signal_gen(n+100+ofs)[ofs:].T
   pred_sig = predict_signal(model, sig[:,:frame_size], shift, n+100)
   pred_sig_arnn = predict_ar_model(arnn_model, sig[:,:frame_size], n+100)
   fig, ax = pl.subplots(5,1)
   ax[0].imshow(log(.1 + abs(sig[:n])), aspect='auto', cmap='gray')
   ax[1].imshow(log(.1 + abs(pred_sig[:n])), aspect='auto', cmap='gray')
   ax[2].imshow(log(.1 + abs(sig[:,:n]-pred_sig[:,:n])), aspect='auto', cmap='gray')
   ax[3].imshow(log(.1 + abs(pred_sig_arnn[:n])), aspect='auto', cmap='gray')
   ax[4].imshow(log(.1 + abs(sig[:,:n]-pred_sig_arnn[:,:n])), aspect='auto', cmap='gray')

def plot_prediction(n=2000, signal_gen=make_signal, k=int(n_nodes/2)):
   sig = signal_gen(n+100).T
   pred_sig = predict_signal(model, sig[:,:frame_size], shift, n+100)
   figure()
   pl.plot(sig[k,:], 'k')
   pl.plot(pred_sig[k,:], 'g')
   pl.plot([frame_size, frame_size],[0,2], '--r', linewidth=2)

def plot_prediction_arnn(n=2000, signal_gen=make_signal, k=int(n_nodes/2)):
   sig = signal_gen(n+100).T
   pred_sig = predict_ar_model(arnn_model, sig[:,:frame_size], n+100)
   figure()
   pl.plot(sig[k,:], 'k')
   pl.plot(pred_sig[k,:], 'g')
   pl.plot([frame_size, frame_size],[0,2], '--r', linewidth=2)


def prediction_dist(num_pred=10, pred_frames=5):
   pred_len = frame_size * pred_frames
   sig = make_signal(num_pred*frame_size + pred_len).T
   diffs = np.zeros_like(sig[:,:pred_len])
   for n in np.arange(num_pred):
      frame = sig[:,n*frame_size:(n+1)*frame_size]
      sig_pred = predict_signal(model, frame, shift, pred_len)[:pred_len]
      diffs = np.concatenate([ diffs, sig_pred[:,:pred_len] - sig[:,n*frame_size:n*frame_size+pred_len] ], axis=0)
   return np.std(diffs,axis=0)


plot_prediction_im(3000, make_signal)
