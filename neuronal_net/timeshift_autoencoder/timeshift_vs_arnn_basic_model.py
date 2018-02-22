#
# simple comparison of basic models
#
# OBSERVATIONS
#  ✔ double prediction regularizer improves performance (sometimes slightly, sometimes more, see below)
#  ✔ TAE is superior especially for slow movements, low frequencies
#  • decaying signals works good on pure linear signal (decaying exponentials)
#    → network probably works in its linear domain
#  • decay on nonlinear signal is very hard to learn

#
# TODOS
#  ✔ non-stationary (decaying) signals
#  ✔ compare against probalistic ARNN (PARNN)
#  • deeper
#  • compare noisy signals
#  • wavelet-orbit

from imports import *
from predictors import *
from metrics import *
from models import *
from keras_tools.wavfile import *
import pylab as pl
import sounddevice as sd

frame_size = 128
shift = 8
n_latent = 4
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

#signal_gen = lambda n: np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#signal_gen = lambda n: np.sin(0.8*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#signal_gen = lambda n: tools.add_noise( np.sin(0.1*np.arange(n) + 6*np.sin(0.017*np.arange(n))), 0.1)
#signal_gen = lambda n: np.exp(-0.001*np.arange(n)) * np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#n_pairs = 2000
#n_epochs = 100
#shift = 4

#wav = loadwav("7.wav")[390:,0]
#wav = loadwav("SDHIT04.WAV")[:,0]
#signal_gen = lambda n: wav[:n]
#frame_size = 128
#n_latent = 24
#shift = 3


loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

arnn_model, parnn_model, tae_model, _ = models({
   "frame_size": frame_size,
   "shift": shift,
   "n_latent": n_latent,
   "in_noise_stddev": in_noise_stddev,
   "code_noise_stddev": code_noise_stddev,
})


def predict_par_model(model, start_frame, n_samples):
   frame_size = start_frame.shape[-1]
   result = start_frame
   frame = start_frame
   for _ in range(n_samples):
      result = np.concatenate([result, [p2x(model.predict(frame.reshape(1,-1))[0])] ])
      frame = result[-frame_size:]
   return result



in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(
   signal_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

next_samples_p = np.array([x2p(x) for x in next_samples])


arnn = arnn_model(in_frames[0])
parnn = parnn_model(in_frames[0])
tae, *_ = tae_model(in_frames[0])
tae21, tae22, *_ = tae_model(in_frames[0])

parnn_metrics = Metrics(fun.bind(predict_par_model, arnn, start_frame=in_frames[0], n_samples=2048))
arnn_metrics = Metrics(fun.bind(predict_ar_model, arnn, start_frame=in_frames[0], n_samples=2048))
tae_metrics = Metrics(fun.bind(predict_signal, tae, start_frame=in_frames[0], shift=shift, n_samples=2048))
tae2_metrics = Metrics(fun.bind(predict_signal, tae21, start_frame=in_frames[0], shift=shift, n_samples=2048))


def train_model(model, ins, outs, metrics_recorder,loss=loss_function):
   model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss)
   tools.train(model, ins, outs, 64, 2*n_epochs, metrics_recorder)
   #model.compile(optimizer=keras.optimizers.SGD(lr=0.005), loss=loss)
   #tools.train(model, ins, outs, 128, n_epochs, metrics_recorder)

train_model(parnn, in_frames, next_samples_p, parnn_metrics, loss=keras.losses.categorical_crossentropy)
train_model(arnn, in_frames, next_samples, arnn_metrics)
train_model(tae, in_frames, out_frames[0], tae_metrics)
train_model(tae22, in_frames, out_frames, tae2_metrics)

sig = signal_gen(4096)
pred_par_sig = predict_ar_model(arnn, in_frames[0], 4096)
pred_ar_sig = predict_ar_model(arnn, in_frames[0], 4096)
pred_sig = predict_signal(tae, in_frames[0], shift, 4096)
pred_sig2 = predict_signal(tae21, in_frames[0], shift, 4096)

def plot_results():
   fig, ax = pl.subplots(4,2)
   ax[0,0].semilogy(tae_metrics.losses, 'b')
   ax[0,0].semilogy(tae2_metrics.losses, 'g')
   ax[0,0].semilogy(arnn_metrics.losses, 'r')
   ax[0,0].semilogy(parnn_metrics.losses, 'c')

   ax[0,1].plot(sig, 'k')
   ax[0,1].plot(tae_metrics.predictions[0], 'b')
   ax[0,1].plot(tae2_metrics.predictions[0], 'g')
   ax[0,1].plot(arnn_metrics.predictions[0], 'r')
   ax[0,1].plot(parnn_metrics.predictions[0], 'c')

   ax[1,0].plot(sig, 'k')
   ax[1,0].plot(tae_metrics.predictions[1], 'b')
   ax[1,0].plot(tae2_metrics.predictions[1], 'g')
   ax[1,0].plot(arnn_metrics.predictions[1], 'r')
   ax[1,0].plot(parnn_metrics.predictions[1], 'c')

   ax[1,1].plot(sig, 'k')
   ax[1,1].plot(tae_metrics.predictions[2], 'b')
   ax[1,1].plot(tae2_metrics.predictions[2], 'g')
   ax[1,1].plot(arnn_metrics.predictions[2], 'r')
   ax[1,1].plot(parnn_metrics.predictions[2], 'c')


   ax[2,0].plot(sig[:-int(15/resample)], sig[int(15/resample):], 'k', linewidth=0.5)
   ax[2,1].plot(pred_sig[:-int(15/resample)], pred_sig[int(15/resample):], 'b', linewidth=0.5)
   ax[3,0].plot(pred_sig2[:-int(15/resample)], pred_sig2[int(15/resample):], 'g', linewidth=0.5)
   ax[3,1].plot(pred_ar_sig[:-int(15/resample)], pred_ar_sig[int(15/resample):], 'r', linewidth=0.5)

plot_results()

# sd.play(sig, 44100)
# sd.play(pred_sig, 44100)