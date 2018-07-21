# RESULT TOOLS
#
# logger
# • 
#
# big result window showing:
# • loss
# • original mixed signal
# • predictions of separations
# • predictions vs original channels
# • predictions error + RMS numbers
# • z-vs-z
# • spec
# • error-rms over time
# • loss: rec per batch → min, max mean curve

import keras
import numpy as np
import matplotlib.pyplot as plt



def windowed(xs, win_size, hop=None):
   if hop == None: hop = win_size
   if win_size <= len(xs):
      for n in range(0, len(xs)-win_size+1, hop):
         yield xs[n:n+win_size]

def build_prediction(model, frames, num=2000):
   pred_frames = model.predict(frames[:num])
   frame_size = len(frames[0])
   times = [np.arange(n, n+frame_size) for n in np.arange(len(pred_frames))]
   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, pred_frames):
      avg_frames[t] += f
   avg_frames *= 1./frame_size
   return avg_frames[:num]

def pred_error(model, frames, gen, n):
   return np.linalg.norm(build_prediction(model, frames, n) - gen(n))


class LossRecorder(keras.callbacks.Callback):

   def __init__(self, **kargs):
      super(LossRecorder, self).__init__(**kargs)
      self.losses = []
      self.grads = []
      self.pred_errors = []

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights

   def on_epoch_end(self, epoch, logs={}):
      self.pred_errors.append(
      [   pred_error(mode1, frames, sig1, 2048)
      ,   pred_error(mode1, frames, sig2, 2048)
      ,   pred_error(mode2, frames, sig2, 2048)
      ,   pred_error(mode2, frames, sig1, 2048)
      ])


# --------------------------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------------------------

def spectrogram(signal, N=256, overlap=0.25):
   hop = int(overlap * N)
   def cos_win(x):
      return x * (0.5 - 0.5*cos(linspace(0,2*pi,len(x))))
   return np.array([ np.abs(np.fft.fft(cos_win(win))[:N//2]) for win in windowed(signal, N, hop) ])

def spec(signal, N=256, overlap=0.25):
   s = spectrogram(signal, N, overlap)
   imshow(log(0.001 + s.T[::-1]), aspect='auto')


def subplots_in(n_rows, n_cols, fig, rect=[0,0,1,1]):

   xs = np.linspace(rect[0], rect[2], n_rows, endpoint=False)
   ys = np.linspace(rect[1], rect[3], n_cols, endpoint=False)
   width = (rect[2] - rect[0]) / n_cols
   hight = (rect[3] - rect[1]) / n_rows

   axs = [[fig.add_axes([x, y, width, hight]) for y in ys] for x in xs]
   return axs


def plot_loss(ax, losses):
   win_size = 100
   min_mean_max = np.array([[w.min(), w.mean(), w.max()] for w in windowed(np.array(losses), win_size)])
   batches = np.arange(min_mean_max.shape[0]) * win_size
   ax.semilogy(batches, min_mean_max[:,0], 'k', linewidth=0.5)
   ax.semilogy(batches, min_mean_max[:,1], 'k', linewidth=1)
   ax.semilogy(batches, min_mean_max[:,2], 'k', linewidth=0.5)
   ax.fill_between(batches, min_mean_max[:,0], min_mean_max[:,2])
   # mean_std = np.array([[w.mean(), w.std()] for w in windowed(np.array(losses), win_size)])
   # batches = np.arange(mean_std.shape[0]) * win_size
   # ax.semilogy(batches, mean_std[:,0], 'k', linewidth=1)
   # ax.semilogy(batches, mean_std[:,0] + mean_std[:,1], 'k', linewidth=0.5)
   # ax.semilogy(batches, mean_std[:,0] - mean_std[:,1], 'k', linewidth=0.5)
   # ax.fill_between(batches, mean_std[:,0] - mean_std[:,1], mean_std[:,0] + mean_std[:,1])


def plot_joint_dist(frames, encoder, fig, rect):
   code = encoder.predict(frames)
   z_dim = code.shape[-1]
   axs = subplots_in(z_dim, z_dim, fig, rect)
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.1)
         ax.axis('off')

   for a in axs[0]: a.set_xlabel(r'$z_n$')


def training_summary(model, mode1, mode2, encoder, gen, sig1, sig2, frames, loss_recorder):

   fig = plt.figure(figsize=(8,8))

   ax = fig.add_axes([0.05, 0.7, 0.4, 0.25])
   plot_loss(ax, loss_recorder.losses)
   plt.title('loss')

   ax = fig.add_axes([0.05, 0.4, 0.4, 0.22])
   ax.plot(gen(2000), 'k', linewidth=0.8)
   plt.title('input')

   ax = fig.add_axes([0.05, 0.1, 0.4, 0.25])
   ax.plot(build_prediction(mode1, frames, 2000), 'k', linewidth=.8)
   ax.plot(build_prediction(mode2, frames, 2000), 'b', linewidth=.8)

   ax = fig.add_axes([0.55, 0.7, 0.4, 0.25])
   pes = loss_recorder.pred_errors
   epochs = np.arange(len(pes))
   ax.semilogy(epochs, pes, 'k')
   plt.title('reconstruction error')

   plot_joint_dist(frames, encoder, fig, [0.5,0.05,0.95,0.5])














def xcorr(m1, m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x2 ) / sqrt( mean(x1**2) * mean(x2**2) )

def nlcorr(m1, m2):
   x1 = m1 - mean(tanh(m1))
   x2 = m2 - mean(tanh(m2))
   return mean( x1 * x2 ) / sqrt( mean(x1**2) * mean(x2**2) )

def xcorr12(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x2 * x2 ) / (sqrt(mean(x1**2)) * mean(x2**3)**(1/3))

def xcorr21(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x1 * x2 ) / (sqrt(mean(x2**2)) * mean(x1**3)**(1/3))

def xcorr22(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x1 * x2 * x2 ) / (sqrt(mean(x2**2)) * mean(x1**2)**(1/2))

def xcorr33(m1,m2):
   x1 = m1 - mean(m1)
   x2 = m2 - mean(m2)
   return mean( x1 * x1 * x1 * x2 * x2 * x2) / sqrt( mean(x2**2) * mean(x1**2) )

def print_corr(X, fxcorr):
   np.set_printoptions(precision=3, suppress=True)
   print(np.array([[xcorr12(c1, c2) for c1 in code.T] for c2 in code.T]))
