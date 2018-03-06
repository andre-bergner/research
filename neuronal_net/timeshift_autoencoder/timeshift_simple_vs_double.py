#
# simple comparison of simple trained vs double-prediction regularization
#
# OBSERVATIONS
#  ✔ double prediction regularizer improves performance (sometimes slightly, sometimes more, see below)
#
# TODOS

from imports import *
from metrics import *
from models import *
import pylab as pl

frame_size = 128
shift = 8
n_latent = 3
in_noise_stddev = 0.05
code_noise_stddev = 0.01
n_pairs = 10000
n_epochs = 40
resample = 1

signal_gen = lambda n: TS.lorenz(resample*n)[::resample]


#### experimenting with resampling the signal and scaling the model window accordingly
# resample = 4
# frame_size = 32
# shift = 2

#signal_gen = lambda n: np.sin(0.05*np.arange(n)) + 0.3*np.sin(0.2212*np.arange(n))
#n_latent = 8

def ae_tae_para_model(example_frame):
   # This model consists of:
   #  • two parallel normal auto encoder
   #  • a mapping from the two successive latent spaces,
   #    thus learning the map into the future within the embedding
   # This model learns a proper auto encoder and a mapping.

   activation = fun.bind(XL.tanhx, alpha=0.1)

   act = lambda: L.Activation(activation)
   eta1 = lambda: F.noise(in_noise_stddev)
   eta2 = lambda: F.noise(code_noise_stddev)

   frame_size = np.size(example_frame)
   x1 = F.input_like(example_frame)
   x2 = F.input_like(example_frame)

   enc1 = F.dense([int(frame_size/2)]) >> act()
   enc2 = F.dense([int(frame_size/4)]) >> act()
   enc3 = F.dense([n_latent]) >> act()
   dec3 = F.dense([int(frame_size/2)]) >> act()
   dec2 = F.dense([int(frame_size/4)]) >> act()
   dec1 = F.dense([frame_size]) #>> act()

   phi1 = F.dense([2*n_latent]) >> act()
   phi2 = F.dense([n_latent]) >> act()

   encoder = eta1() >> enc1 >> enc2 >> enc3
   decoder = eta2() >> dec3 >> dec2 >> dec1
   shift = phi1 >> phi2
   ae_chain = encoder >> decoder

   return (
      M.Model([x1,x2], [ae_chain(x1), ae_chain(x2), (encoder >> shift >> decoder)(x1)]),
      M.Model([x1], [(encoder >> shift >> decoder)(x1)]),
      M.Model([x1], [encoder(x1)]),
      # M.Model([x1], [shift(x1)]),
   )


loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

_, _, tae_model, _ = models({
   "frame_size": frame_size,
   "shift": shift,
   "n_latent": n_latent,
   "in_noise_stddev": in_noise_stddev,
   "code_noise_stddev": code_noise_stddev,
})

in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(
   signal_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

ae_tae_train, ae_tae, ae_tae_encoder = ae_tae_para_model(in_frames[0])

tae, *_ = tae_model(in_frames[0])
tae21, tae22, *_ = tae_model(in_frames[0])

tae_metrics = Metrics(fun.bind(predict_signal, tae, start_frame=in_frames[0], shift=shift, n_samples=2048))
tae2_metrics = Metrics(fun.bind(predict_signal, tae21, start_frame=in_frames[0], shift=shift, n_samples=2048))
ae_tae_metrics = Metrics(fun.bind(predict_signal, ae_tae, start_frame=in_frames[0], shift=shift, n_samples=2048))


def train_model(model, ins, outs, metrics_recorder, loss=loss_function, n_epochs=n_epochs):
   model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss)
   tools.train(model, ins, outs, 64, 2*n_epochs, metrics_recorder)
   #model.compile(optimizer=keras.optimizers.SGD(lr=0.005), loss=loss)
   #tools.train(model, ins, outs, 128, n_epochs, metrics_recorder)

#train_model(ae_tae_train, [in_frames, out_frames[0]], [in_frames, out_frames[0], out_frames[0]], ae_tae_metrics, n_epochs=2*n_epochs)
train_model(ae_tae_train, [in_frames, out_frames[0]], [in_frames, out_frames[0], out_frames[0]], ae_tae_metrics)
train_model(tae, in_frames, out_frames[0], tae_metrics)
train_model(tae22, in_frames, out_frames, tae2_metrics)

sig = signal_gen(4096)
pred_sig = predict_signal(tae, in_frames[0], shift, 4096)
pred_sig2 = predict_signal(tae21, in_frames[0], shift, 4096)
pred_sig3 = predict_signal(ae_tae, in_frames[0], shift, 4096)

def plot_results():
   fig, ax = pl.subplots(4,2)
   ax[0,0].semilogy(tae_metrics.losses, 'b')
   ax[0,0].semilogy(tae2_metrics.losses, 'g')
   ax[0,0].semilogy(ae_tae_metrics.losses, 'r')

   ax[0,1].plot(sig, 'k')
   ax[0,1].plot(tae_metrics.predictions[0], 'b')
   ax[0,1].plot(tae2_metrics.predictions[0], 'g')
   ax[0,1].plot(ae_tae_metrics.predictions[0], 'r')

   ax[1,0].plot(sig, 'k')
   ax[1,0].plot(tae_metrics.predictions[1], 'b')
   ax[1,0].plot(tae2_metrics.predictions[1], 'g')
   ax[1,0].plot(ae_tae_metrics.predictions[1], 'r')

   ax[1,1].plot(sig, 'k')
   ax[1,1].plot(tae_metrics.predictions[2], 'b')
   ax[1,1].plot(tae2_metrics.predictions[2], 'g')
   ax[1,1].plot(ae_tae_metrics.predictions[2], 'r')


   ax[2,0].plot(sig[:-int(15/resample)], sig[int(15/resample):], 'k', linewidth=0.5)
   ax[2,1].plot(pred_sig[:-int(15/resample)], pred_sig[int(15/resample):], 'b', linewidth=0.5)
   ax[3,0].plot(pred_sig2[:-int(15/resample)], pred_sig2[int(15/resample):], 'g', linewidth=0.5)
   ax[3,1].plot(pred_sig3[:-int(15/resample)], pred_sig3[int(15/resample):], 'r', linewidth=0.5)

plot_results()
