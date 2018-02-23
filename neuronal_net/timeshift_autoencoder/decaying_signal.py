from imports import *
from models import *
from metrics import *
from keras_tools.wavfile import *
import pylab as pl

#signal_gen = lambda n: np.exp(-0.002*np.arange(n)) * np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#signal_gen = lambda n: np.linspace(1,0,n) * np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
signal_gen = lambda n: np.sin(0.008*np.arange(n)) * np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))
#signal_gen = lambda n: .5*(.8+.2*np.cos(0.008*np.arange(n))) * np.sin(0.2*np.arange(n) + 6*np.sin(0.017*np.arange(n)))

frame_size = 256
shift = 128
n_latent = 8
in_noise_stddev = 0.05
code_noise_stddev = 0.01
n_pairs = 2000
n_epochs = 300



activation = fun.bind(XL.tanhx, alpha=0.1)

loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)


act = lambda: L.Activation(activation)
eta1 = lambda: F.noise(in_noise_stddev)
eta2 = lambda: F.noise(code_noise_stddev)

def arnn_model(example_frame):
   x = F.input_like(example_frame)

   d1 = F.dense([int(frame_size/2)]) >> act()
   d2 = F.dense([int(frame_size/4)]) >> act()
   d3 = F.dense([n_latent]) >> act()
   d4 = F.dense([int(frame_size/2)]) >> act()
   d5 = F.dense([int(frame_size/4)]) >> act()
   d6 = F.dense([1]) #>> act()

   chain = eta1() >> d1 >> d2 >> d3 >> eta2() >> d4 >> d5 >> d6

   return M.Model([x], [chain(x)])


def tae_model(example_frame):
   frame_size = np.size(example_frame)
   x = F.input_like(example_frame)

   enc1 = F.dense([int(frame_size/2)]) >> act()
   enc2 = F.dense([int(frame_size/4)]) >> act() >> F.dropout(0.4) # >> F.batch_norm() >> F.dropout(0.2)
   enc3 = F.dense([int(frame_size/8)]) >> act() >> F.dropout(0.4) # >> F.batch_norm() >> F.dropout(0.2)
   enc4 = F.dense([n_latent]) >> act()
   dec4 = F.dense([int(frame_size/8)]) >> act() >> F.dropout(0.4) #>> F.batch_norm() >> F.dropout(0.2)
   dec3 = F.dense([int(frame_size/2)]) >> act() >> F.dropout(0.4) #>> F.batch_norm() >> F.dropout(0.2)
   dec2 = F.dense([int(frame_size/4)]) >> act()
   dec1 = F.dense([frame_size]) #>> act()

   encoder = enc1 >> enc2 >> enc3 >> enc4
   decoder = dec4 >> dec3 >> dec2 >> dec1
   y = eta1() >> encoder >> eta2() >> decoder
   latent = encoder(x)
   out = decoder(latent)

   return M.Model([x], [out]), M.Model([x], [out, y(out)]), M.Model([x], [latent])#, XL.jacobian(latent,x)


def normalize_data(x, y):
   moments = [(np.mean(s), np.std(s)) for s in x]
   x = np.array([(p-m)/d for p,(m,d) in zip(x,moments)])
   y = np.array([(p-m)/d for p,(m,d) in zip(y,moments)])
   return x, y


in_frames, out_frames, next_samples, next_samples2 = TS.make_training_set(
   signal_gen, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)


arnn = arnn_model(in_frames[0])
tae, *_ = tae_model(in_frames[0])
tae21, tae22, *_ = tae_model(in_frames[0])


def normalizing_predict(model):

   class PredictNomarlizer:

      def __init__(self, model):
         self.model = model

      def __getattr__(self, name):
         return getattr(self.model, name)

      def predict(self, x, *args, **kwargs):
         moments = [(np.mean(s), np.std(s)) for s in x]
         x = np.array([(p-m)/d for p,(m,d) in zip(x,moments)])
         y = model.predict(x, *args, **kwargs)
         y = np.array([q*d+m for q,(m,d) in zip(y,moments)])
         return y

   return PredictNomarlizer(model)

taen = normalizing_predict(tae)

predict_signal = predict_signal2

arnn_metrics = Metrics(fun.bind(predict_ar_model, arnn, start_frame=in_frames[0], n_samples=2048))
#tae_metrics = Metrics(fun.bind(predict_signal, tae, start_frame=in_frames[0], shift=shift, n_samples=2048))
tae_metrics = Metrics(fun.bind(predict_signal, taen, start_frame=in_frames[0], shift=shift, n_samples=2048))
tae2_metrics = Metrics(fun.bind(predict_signal, tae21, start_frame=in_frames[0], shift=shift, n_samples=2048))


def train_model(model, ins, outs, metrics_recorder):
   model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
   tools.train(model, ins, outs, 32, n_epochs, metrics_recorder)
   tools.train(model, ins, outs, 64, n_epochs, metrics_recorder)
   model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
   tools.train(model, ins, outs, 64, n_epochs, metrics_recorder)
   #model.compile(optimizer=keras.optimizers.SGD(lr=0.005), loss=loss_function)
   #tools.train(model, ins, outs, 128, n_epochs, metrics_recorder)

#train_model(arnn, in_frames, next_samples, arnn_metrics)
#train_model(tae, in_frames, out_frames[0], tae_metrics)
train_model(tae, *normalize_data(in_frames, out_frames[0]), tae_metrics)
#train_model(tae22, in_frames, out_frames, tae2_metrics)

sig = signal_gen(2048)
#pred_ar_sig = predict_ar_model(arnn, in_frames[0], 2048)
pred_sig = predict_signal(taen, in_frames[0], shift, 2048)
#pred_sig2 = predict_signal(tae21, in_frames[0], shift, 2048)

def plot_results():
   fig, ax = pl.subplots(2,2)
   ax[0,0].semilogy(tae_metrics.losses, 'b')
   #ax[0,0].semilogy(tae2_metrics.losses, 'g')
   #ax[0,0].semilogy(arnn_metrics.losses, 'r')

   ax[0,1].plot(sig, 'k')
   ax[0,1].plot(tae_metrics.predictions[0], 'b')
   #ax[0,1].plot(tae2_metrics.predictions[0], 'g')
   #ax[0,1].plot(arnn_metrics.predictions[0], 'r')

   ax[1,0].plot(sig, 'k')
   ax[1,0].plot(tae_metrics.predictions[1], 'b')
   #ax[1,0].plot(tae2_metrics.predictions[1], 'g')
   #ax[1,0].plot(arnn_metrics.predictions[1], 'r')

   ax[1,1].plot(sig, 'k')
   ax[1,1].plot(tae_metrics.predictions[2], 'b')
   #ax[1,1].plot(tae2_metrics.predictions[2], 'g')
   #ax[1,1].plot(arnn_metrics.predictions[2], 'r')

plot_results()


#plot_top_and_worst = fun.bind(tools.plot_top_and_worst, tae21, in_frames, out_frames[0])
plot_top_and_worst = fun.bind(tools.plot_top_and_worst, taen, in_frames, out_frames[0])
plot_top_and_worst()
