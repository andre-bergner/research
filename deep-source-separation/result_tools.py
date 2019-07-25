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

def all_predictions(model, frames):
   """Returns all parallel prediction of per sample."""
   pred = model.predict(frames)
   N = pred.shape[-1]
   return np.array([pred[N-n:-n-1,n] for n in range(N)]).T

def build_prediction(model, frames, num=None, stride=1):
   pred_frames = model.predict(frames[:num])
   frame_size = len(frames[0])
   #times = [np.arange(n, n+frame_size) for n in np.arange(len(pred_frames))]
   times = [np.arange(n, n+stride*frame_size, stride) for n in np.arange(len(pred_frames))]
   avg_frames = np.zeros(times[-1][-1]+1)
   for t, f in zip(times, pred_frames):
      avg_frames[t] += f
   #fade = np.arange(frame_size) + 1
   #denom = np.concatenate([fade, frame_size * np.ones(len(avg_frames) - 2*frame_size), fade[::-1]])
   fade = np.repeat(np.arange(frame_size) + 1, stride)
   denom = np.concatenate([fade, frame_size * np.ones(len(avg_frames) - 2 * stride * frame_size), fade[::-1]])
   avg_frames *= 1. / denom
   return avg_frames[:num]

def pred_error(model, frames, gen, n):
   x = gen(n)
   # y = build_prediction(model, frames, n)
   y = model.infer(n)
   return np.linalg.norm(y-x)**2 / (np.linalg.norm(x) * np.linalg.norm(y))


class LossRecorder(keras.callbacks.Callback):

   # TODO record loss per epoch: mean, median, min, max, std

   def __init__(self, **kargs):
      super(LossRecorder, self).__init__(**kargs)
      self.losses = []
      self.grads = []

   def _current_weights(self):
      return [l.get_weights() for l in self.model.layers if len(l.get_weights()) > 0]

   def on_train_begin(self, logs={}):
      self.last_weights = self._current_weights()

   def on_batch_end(self, batch, logs={}):
      self.losses.append(logs.get('loss'))
      new_weights = self._current_weights()
      self.grads.append([ (w2[0]-w1[0]).mean() for w1,w2 in zip(self.last_weights, new_weights) ])
      self.last_weights = new_weights


# --------------------------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------------------------

def spectrogram(signal, N=256, overlap=0.25):
   hop = int(overlap * N)
   def cos_win(x):
      return x * (0.5 - 0.5*np.cos(np.linspace(0,2*np.pi,len(x))))
   return np.array([ np.abs(np.fft.fft(cos_win(win))[:N//2]) for win in windowed(signal, N, hop) ])

def spec(signal, N=256, overlap=0.25):
   s = spectrogram(signal, N, overlap)
   imshow(log(0.001 + s.T[::-1]), aspect='auto')


def subplots_in(n_rows, n_cols, fig, rect=[0,0,1,1]):

   xs = np.linspace(rect[0], rect[2], n_rows, endpoint=False)
   ys = np.linspace(rect[1], rect[3], n_cols, endpoint=False)
   width = (rect[2] - rect[0]) / n_cols
   hight = (rect[3] - rect[1]) / n_rows

   axs = [[fig.add_axes([x, y, 0.95*width, 0.95*hight]) for y in ys] for x in xs]
   return axs


def plot_loss(ax, losses):
   win_size = 100
   min_mean_max = np.array([[w.min(), w.mean(), w.max()] for w in windowed(np.array(losses), win_size)])
   batches = np.arange(min_mean_max.shape[0]) * win_size
   ax.semilogy(batches, min_mean_max[:,0], 'k', linewidth=0.5)
   ax.semilogy(batches, min_mean_max[:,1], 'k', linewidth=1)
   ax.semilogy(batches, min_mean_max[:,2], 'k', linewidth=0.5)
   ax.fill_between(batches, min_mean_max[:,0], min_mean_max[:,2], alpha=0.5)
   # mean_std = np.array([[w.mean(), w.std()] for w in windowed(np.array(losses), win_size)])
   # batches = np.arange(mean_std.shape[0]) * win_size
   # ax.semilogy(batches, mean_std[:,0], 'k', linewidth=1)
   # ax.semilogy(batches, mean_std[:,0] + mean_std[:,1], 'k', linewidth=0.5)
   # ax.semilogy(batches, mean_std[:,0] - mean_std[:,1], 'k', linewidth=0.5)
   # ax.fill_between(batches, mean_std[:,0] - mean_std[:,1], mean_std[:,0] + mean_std[:,1])


def plot_joint_dist(frames, encoder, fig, rect):
   frames = frames[:min(8000, len(frames))]
   code = encoder.predict(frames)
   z_dim = code.shape[-1]
   axs = subplots_in(z_dim, z_dim, fig, rect)
   for ax_rows, c1 in zip(axs, code.T):
      for ax, c2 in zip(ax_rows, code.T):
         ax.plot( c2, c1, '.k', markersize=0.1)
         ax.axis('off')

   for a in axs[0]: a.set_xlabel(r'$z_n$')



def plot_histogramm_in(ax, data, num_bins=50, range=(-1, 1), color='k'):
    hist, bins = np.histogram(data, num_bins, range=range)
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    norm = 0.5 * len(bins) / len(data)
    ax.bar(center, norm*hist, align='center', width=width, edgecolor=None, color=color)


from scipy.stats import gaussian_kde

def plot_kde_in(ax, data, range=(-1, 1), width=0.1, color='#222222', alpha=1.0, swap_axes=False):
    kde = gaussian_kde(data, width)
    num = 100
    xs = np.linspace(*range, num)
    ys = kde.evaluate(xs)
    #ax.plot(xs, ys)
    if swap_axes:
        ax.fill_betweenx(xs, np.zeros(num), ys, color=color, alpha=alpha, zorder=10)
    else:
        ax.fill_between(xs, np.zeros(num), ys, color=color, alpha=alpha, zorder=10)


def estimate_kde2d(x, y, width=0.05):
   xmin = x.min()
   xmax = x.max()
   ymin = y.min()
   ymax = y.max()
   X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
   positions = np.vstack([X.ravel(), Y.ravel()])
   values = np.vstack([x, y])
   kernel = gaussian_kde(values, width)
   kde = np.reshape(kernel.evaluate(positions).T, X.shape)
   return kde


def scatter_img(xs, ys, bins=100):
    img = np.ones((bins+2, bins+2))
    x_min = xs.min()
    y_min = ys.min()
    x_rng = xs.max() - x_min
    y_rng = ys.max() - y_min

    map = lambda x, y: (int((bins-1)*(x-x_min)/x_rng)+1, int((bins-1)*(y-y_min)/y_rng)+1)

    for (x, y) in zip(xs, ys):
        n, m = map(x,y)
        img[n, m] *= 0.93
        img[n-1, m] *= 0.97
        img[n+1, m] *= 0.97
        img[n, m-1] *= 0.97
        img[n, m+1] *= 0.97
        img[n-1, m+1] *= 0.995
        img[n-1, m-1] *= 0.995
        img[n+1, m+1] *= 0.995
        img[n+1, m-1] *= 0.995

    return 1 - img[1:-1, 1:-1]



def plot_latent_space_impl(
   fig, rect, frames, encoder,
   with_labels=True, with_marginals=True, max_num_frames=None, method='hist2', bins=70, pow=1
):

    def scaled_log(x):
       return np.log(log_ofs + (x - x.min())/(x.max()-x.min()))

    plt.rcParams['mathtext.fontset'] = 'stix' # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

    if max_num_frames:
       frames = frames[:min(max_num_frames, len(frames))]
    code = encoder.predict(frames)
    z_dim = code.shape[-1]
    if with_marginals:
       axs = subplots_in(z_dim+1, z_dim+1, fig, rect)
    else:
       axs = subplots_in(z_dim, z_dim, fig, rect)

    def get_range(z):
       z_min = z.min()
       z_max = z.max()
       z_diff = z_max - z_min
       return [z_min - 0.1*z_diff, z_max + 0.1*z_diff]

    for ax_rows in axs:
       for ax in ax_rows:
           ax.axis('off')

    # -----------------------------------------------------------
    # plotting the joint densities

    if with_marginals:
       axss = axs[:-1]
    else:
       axss = axs
    for ax_rows, c1 in zip(axss, code.T):
       if with_marginals:
          axsss = ax_rows[:-1]
       else:
          axsss = ax_rows
       for ax, c2 in zip(axsss, code.T):
           if method == 'hist':
              hist, *_ = np.histogram2d(c1, c2, bins=bins, range=[get_range(c1), get_range(c2)])
              hist = np.rot90(hist)
              ax.imshow(1-hist**pow, cmap='gray', extent=[0,1,0,1])
           elif method == 'hist2':
              hist = scatter_img(c1, c2, bins=bins)
              hist = np.rot90(hist)
              ax.imshow(1-hist**pow, cmap='gray', extent=[0,1,0,1])
           elif method == 'kde':
              if (c1 == c2).all():
                 continue
              kde = estimate_kde2d(c1, c2)
              kde = np.rot90(kde)
              ax.imshow(1-kde, cmap='gray', extent=[0,1,0,1])
           else: # default: 'scatter'
              ax.plot( c1, c2, '.k', markersize=0.1)
              ax.set_xlim(get_range(c1))
              ax.set_ylim(get_range(c2))

    # -----------------------------------------------------------
    # plotting the marginals

    if with_marginals:

       for ax, z in zip(axs, code.T):
           range = get_range(z)
           plot_kde_in(ax[-1], z, range=range)
           ax[-1].set_xlim(range)
 
       for ax, z in zip(axs[-1], code.T):
           range = get_range(z)
           plot_kde_in(ax, z, range=range, swap_axes=True)
           ax.set_ylim(range)

    # -----------------------------------------------------------
    # plotting the labels

    if with_labels == False:
       return

    for n, ax in enumerate(axs[0][:-1], 1):
        pos = ax.get_position()
        dx = pos.x1 - pos.x0
        dy = pos.y1 - pos.y0
        fig.text(pos.x0 - 0.06, pos.y0 + 0.4*dy, r'$z_{{{}}}$'.format(n), fontsize=16)

    for n, ax in enumerate(axs[:-1], 1):
        pos = ax[0].get_position()
        dx = pos.x1 - pos.x0
        dy = pos.y1 - pos.y0
        fig.text(pos.x0 + 0.3*dx, pos.y0 - 0.4*dy, r'$z_{{{}}}$'.format(n), fontsize=16)


def plot_latent_space(sep, **kwargs):
    fig = plt.figure(figsize=(5, 5))
    return plot_latent_space_impl(fig, [0.1, 0.1, 0.95, 0.95], sep.frames, sep.encoder, **kwargs)


def training_summary_impl(model, mode1, mode2, encoder, gen, sig1, sig2, frames, recorder, **kwargs):

   fig = plt.figure(figsize=(8,8))

   ax = fig.add_axes([0.05, 0.7, 0.4, 0.25])
   loss = np.array(recorder.losses)
   for l in loss.T:
      plot_loss(ax, l)
   plt.title('loss')

   ax = fig.add_axes([0.05, 0.4, 0.4, 0.22])
   ax.plot(gen(2000), 'k', linewidth=0.8)
   plt.title('input')

   ax = fig.add_axes([0.05, 0.1, 0.4, 0.25])
   ax.plot(sig1(2000), 'k', linewidth=.8)
   ax.plot(sig2(2000), 'k', linewidth=.8)
   # ax.plot(build_prediction(mode1, frames, 2000), 'r')
   # ax.plot(build_prediction(mode2, frames, 2000), 'b')
   ax.plot(mode1.infer(2000), 'r')
   ax.plot(mode2.infer(2000), 'b')

   ax = fig.add_axes([0.5, 0.7, 0.4, 0.25])
   pes = recorder.pred_errors
   epochs = np.arange(len(pes))
   try:
      ax.semilogy(epochs, pes, 'k')
   except:
      pass
   ax.spines['top'].set_alpha(0.0)
   ax2 = ax.twinx()
   try:
      ax2.semilogy(epochs, recorder.mutual_information, color='tab:blue')
   except:
      pass
   ax2.spines['top'].set_alpha(0.0)
   ax2.spines['right'].set_color('tab:blue')
   ax2.tick_params(axis='y', labelcolor='tab:blue')
   ax2.yaxis.label.set_color('tab:blue')
   plt.title('reconstruction error')

   #plot_joint_dist(frames, encoder, fig, [0.5,0.05,0.95,0.5])
   plot_latent_space_impl(fig, [0.5,0.05,0.95,0.5], frames, encoder, **kwargs)


def training_summary(sep, **kwargs):
   signal = sep.signal_gens[0] + sep.signal_gens[1]
   recordings = type('', (), {})()  # duck typing: stackoverflow.com/questions/19476816/creating-an-empty-object-in-python
   recordings.losses = sep.loss_recorder.losses
   recordings.pred_errors = sep.sep_recorder.pred_errors
   recordings.mutual_information = sep.sep_recorder.mutual_information
   return training_summary_impl(
      sep.model, *sep.modes, sep.encoder, signal,
      *sep.signal_gens, sep.frames, recordings,
      **kwargs
   )

