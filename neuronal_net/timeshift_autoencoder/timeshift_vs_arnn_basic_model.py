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
n_epochs = 10
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



in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(
   signal_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)

next_samples_p = np.array([x2p(x) for x in next_samples])

make_test_signal = lambda n: TS.lorenz(n, [0,1,0])

def prediction_dist(predictor, num_pred=10, pred_frames=5):
   pred_len = frame_size * pred_frames
   sig = make_test_signal(num_pred*frame_size + pred_len)
   diffs = []
   for n in np.arange(num_pred):
      frame = sig[n*frame_size:(n+1)*frame_size].copy()
      sig_pred = predictor(frame, pred_len)[:pred_len]
      diffs.append(sig_pred - sig[n*frame_size:n*frame_size+pred_len])
   return np.std(np.array(diffs), axis=0)



arnn = arnn_model(in_frames[0])
parnn = parnn_model(in_frames[0])
tae, *_ = tae_model(in_frames[0])
tae21, tae22, *_ = tae_model(in_frames[0])

parnn_metrics = Metrics(fun.bind(predict_par_model, arnn, start_frame=in_frames[0], n_samples=2048))
arnn_metrics = Metrics(fun.bind(predict_ar_model, arnn, start_frame=in_frames[0], n_samples=2048))
tae_metrics = Metrics(fun.bind(predict_signal, tae, start_frame=in_frames[0], shift=shift, n_samples=2048))
tae2_metrics = Metrics(fun.bind(predict_signal, tae21, start_frame=in_frames[0], shift=shift, n_samples=2048))

#parnn_metrics = Metrics( fun.bind(prediction_dist, predictor=lambda f,x: predict_par_model(parnn, start_frame=f, n_samples=x)))
#arnn_metrics = Metrics(  fun.bind(prediction_dist, predictor=lambda f,x: predict_ar_model(arnn, start_frame=f, n_samples=x)))
#tae_metrics = Metrics(   fun.bind(prediction_dist, predictor=lambda f,x: predict_signal(tae, start_frame=f, shift=shift, n_samples=x)))
#tae2_metrics = Metrics(  fun.bind(prediction_dist, predictor=lambda f,x: predict_signal(tae21, start_frame=f, shift=shift, n_samples=x)))


def train_model(model, ins, outs, metrics_recorder,loss=loss_function):
   model.compile(optimizer=keras.optimizers.Adam(), loss=loss)
   tools.train(model, ins, outs, 32, 3*n_epochs, metrics_recorder)

train_model(parnn, in_frames, next_samples_p, parnn_metrics, loss=keras.losses.categorical_crossentropy)
train_model(arnn, in_frames, next_samples, arnn_metrics)
train_model(tae, in_frames, out_frames[0], tae_metrics)
train_model(tae22, in_frames, out_frames, tae2_metrics)

sig = signal_gen(4096)
pred_par_sig = predict_par_model(parnn, in_frames[0], 4096)
pred_ar_sig = predict_ar_model(arnn, in_frames[0], 4096)
pred_sig = predict_signal(tae, in_frames[0], shift, 4096)
pred_sig2 = predict_signal(tae21, in_frames[0], shift, 4096)

def plot_results():
   fig, ax = pl.subplots(3,2)
   ax[0,0].semilogy(tae_metrics.losses, 'b')
   ax[0,0].semilogy(tae2_metrics.losses, 'g')
   ax[0,0].semilogy(arnn_metrics.losses, 'r')
   ax[0,0].semilogy(parnn_metrics.losses, 'c')

   ax[0,1].plot(sig[:-int(15/resample)], sig[int(15/resample):], 'k', linewidth=0.5)
   ax[1,0].plot(pred_sig[:-int(15/resample)], pred_sig[int(15/resample):], 'b', linewidth=0.5)
   ax[1,1].plot(pred_sig2[:-int(15/resample)], pred_sig2[int(15/resample):], 'g', linewidth=0.5)
   ax[2,0].plot(pred_ar_sig[:-int(15/resample)], pred_ar_sig[int(15/resample):], 'r', linewidth=0.5)
   ax[2,1].plot(pred_par_sig[:-int(15/resample)], pred_par_sig[int(15/resample):], 'c', linewidth=0.5)

plot_results()
