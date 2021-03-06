from imports import *
from predictors import *
#from test_signals import *
import pylab as pl
from pylab import *
from matplotlib.pyplot import imread

# ------------------------------------------------------------------------------
# MODEL PARAMETERS
# ------------------------------------------------------------------------------

kern_len = 5
noise_stddev = 0.03
noise_level = 0.0#1
use_bias = True
n_nodes = 20
n_latent = 20
frame_size = 128
shift = 32
n_pairs = 2000
n_epochs = 50
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
conv1d = lambda feat, stride=1: F.conv1d(int(feat), kern_len, stride=stride, activation=None, use_bias=use_bias)


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


def bwimread(filename):
   img = imread(filename) / 255
   return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3

# img = imread("pattern.jpg") / 260
# img = bwimread("textures/bgfons.com/plastic_texture1343.jpg")
# make_signal = lambda n: img[100:140:2].T


# n_latent = 80
# n_nodes = 40
# img = imread("textures/bgfons.com/rope_texture2078.jpg") / 260
# img_sig = np.concatenate([img[100:140].T, img[140:180].T], axis=0)
# make_signal = lambda n: img_sig[:n]


def part_rope_image(k=32):
   global n_pairs, n_nodes, n_latent, n_epochs, make_signal, loss_function
   n_pairs = 3000
   n_latent = 40
   n_nodes = k
   n_epochs = 20
   k0 = 500
   #img = bwimread("textures/bgfons.com/rope_texture2074.jpg")
   #n_pairs = 1500
   #img = bwimread("textures/adult-anger-angry-34667.jpg")
   #img = bwimread("textures/stockvault-forest-fog142490.jpg")
   img = bwimread("textures/adult-art-black-and-white-604694.jpg")
   make_signal = lambda n: img[k0:k0+kk:4,:n].T
   kk = n_nodes*4
   #make_signal = lambda n: np.concatenate([img[k0:k0+kk:4,:].T, img[k0+kk:k0+2*kk:4,:].T, img[k0+2*kk:k0+3*kk:4,:].T])[:n]

   def make_training_set(n_pairs=n_pairs, frame_size=frame_size, shift=shift, n_out=1):
      max_x = img.shape[0] - frame_size - shift*n_out - 1
      max_y = img.shape[1] - n_nodes - 1
      rnd = np.random.rand
      in_frames = [] 
      out_frames = []
      out_frames2 = []
      for n in range(n_pairs):
         rnd_x = int(rnd() * max_x)
         rnd_y = int(rnd() * max_y)
         in_frames.append(img[rnd_x:rnd_x+frame_size, rnd_y:rnd_y+n_nodes])
         out_frames.append(img[rnd_x+shift:rnd_x+frame_size+shift, rnd_y:rnd_y+n_nodes])
         out_frames2.append(img[rnd_x+2*shift:rnd_x+frame_size+2*shift, rnd_y:rnd_y+n_nodes])
      return np.array(in_frames), [np.array(out_frames), np.array(out_frames2)]

   loss_function = lambda y_true, y_pred: \
      0.5*mae(y_true, y_pred) + \
      mae(diff2(y_true), diff2(y_pred)) + \
      mae(diff2(diff2(y_true)), diff2(diff2(y_pred)))
   return make_training_set

def full_rope_image():
   global n_pairs, n_nodes, n_latent, n_epochs, make_signal, loss_function
   n_pairs = 2000
   n_nodes = 450
   n_latent = 40
   n_epochs = 60
   # n_nodes = 450
   # img = bwimread("textures/bgfons.com/rope_texture2074.jpg")
   # make_signal = lambda n: img[0:1800:4,:n].T
   img = bwimread("textures/adult-art-black-and-white-604694.jpg")
   make_signal = lambda n: img[0::4,:n].T
   n_pairs = img.shape[1] - 2*shift - frame_size
   n_nodes = img.shape[0] // 4
   n_epochs = 120000 // n_pairs  
   loss_function = lambda y_true, y_pred: \
      0.5*mae(y_true, y_pred) + \
      mae(diff2(y_true), diff2(y_pred)) + \
      mae(diff2(diff2(y_true)), diff2(diff2(y_pred)))

# n_pairs = 2000
# n_nodes = 40
# n_latent = 40
# n_epochs = 50
#frame_gen = part_rope_image(64)
full_rope_image()


# ------------------------------------------------------------------------------
# PREPARING DATA
# ------------------------------------------------------------------------------

in_frames, out_frames, next_samples, _ = TS.make_training_set(make_signal, frame_size=frame_size, n_pairs=n_pairs, shift=shift, n_out=2)
#in_frames, out_frames = frame_gen(n_pairs=n_pairs, frame_size=frame_size, shift=shift, n_out=2)
in_frames = in_frames.transpose(0,2,1)
out_frames = [out_frames[0].transpose(0,2,1), out_frames[1].transpose(0,2,1)]
next_samples = next_samples.reshape(-1,n_nodes,1)


# ------------------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------------------

def make_model_2d(example_frame, latent_size, simple=True):
   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   # tp = fun._ >> L.Permute((1, 2))  TODO: try using permute/transpose

   if simple:

      print("simple model")

      encoder = (  conv1d(sig_len/2) >> act() # >> F.dropout(0.2)
                >> conv1d(sig_len/4) >> act() # >> F.dropout(0.2)
                >> F.flatten() >> dense([n_latent]) >> act() # >> F.batch_norm()
                )

      decoder = (  dense([n_nodes, int(sig_len/4)]) >> act() # >> F.batch_norm() >> F.dropout(0.2)
                >> conv1d(sig_len/2) >> act() # >> F.dropout(0.2)
                >> conv1d(sig_len) #>> act()
                )

   else:

      encoder = (  conv1d(sig_len/2) >> act() >> F.dropout(0.2)
                >> conv1d(sig_len/4) >> act() >> F.dropout(0.2)
                >> conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
                >> F.flatten() >> dense([n_latent]) >> act() >> F.batch_norm()
                )

      # TODO: figure out dimension from shape
      decoder = (  dense([n_nodes, int(sig_len/4)]) >> act() >> F.batch_norm() >> F.dropout(0.2)
                >> conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
                >> conv1d(sig_len/2) >> act() >> F.dropout(0.2)
                >> conv1d(sig_len) #>> act()
                )

   y = eta() >> encoder >> eta() >> decoder
   latent = encoder(x)
   return M.Model([x], [y(x)]), M.Model([x], [y(x), y(y(x))]), M.Model([x], [latent])#, XL.jacobian(latent,x)



def make_model_2d_conv2latent(example_frame):
   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   conv1d = lambda feat, stride=1: F.conv1d(int(feat), 7, stride=stride)

   encoder = (  conv1d(sig_len//2, 2) >> act() >> F.dropout(0.2)
             >> conv1d(sig_len//2, 2) >> act() >> F.dropout(0.2)
             >> conv1d(sig_len//2, 2) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> conv1d(sig_len//4, 2) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> conv1d(sig_len//4, 2) >> act() >> F.batch_norm()
             )

   decoder = (  F.up(2) >> conv1d(sig_len/4) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> F.up(2) >> conv1d(sig_len/2) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> F.up(2) >> conv1d(sig_len/2) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> F.up(2) >> conv1d(sig_len/2) >> act() >> F.dropout(0.2)
             >> F.up(2) >> conv1d(sig_len)
             )

   y = eta() >> encoder >> eta() >> decoder
   latent = encoder(x)
   out = decoder(latent)
   return M.Model([x], [y(x)]), M.Model([x], [y(x), y(y(x))]), M.Model([x], [latent])#, XL.jacobian(latent,x)



def make_model_2d_conv(example_frame, latent_size):

   sig_len = example_frame.shape[-1]
   x = F.input_like(example_frame)
   eta = lambda: F.noise(noise_stddev)

   conv = lambda feat, stride=(1,1): F.conv2d(feat, (kern_len, 7), stride=stride)
   up = lambda x,y: F.up2d((x,y))

   encoder = (  F.reshape(list(example_frame.shape) + [1]) 
             >> conv(2, (1,2)) >> act() >> F.dropout(0.2)
             >> conv(2, (1,2)) >> act() >> F.dropout(0.2)
             >> conv(4, (1,2)) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> conv(4, (1,2)) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> F.flatten() >> dense([n_latent]) >> act() >> F.batch_norm()
             )

   decoder = (  dense([n_nodes, sig_len//16, 4]) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> up(1,2) >> conv(4) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> up(1,2) >> conv(2) >> act() >> F.batch_norm() >> F.dropout(0.2)
             >> up(1,2) >> conv(2) >> act() >> F.dropout(0.2)
             >> up(1,2) >> conv(1) #>> act()
             >> F.reshape(example_frame.shape)
             )

   y = eta() >> encoder >> eta() >> decoder
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

      y = d1 >> d2 >> d3 >> d4 >> d5
      return M.Model([x], [y(x)])

   else:

      d1 = conv1d(sig_len/2) >> act() >> F.dropout(0.2)
      d2 = conv1d(sig_len/4) >> act() >> F.dropout(0.2)
      d3 = conv1d(sig_len/8) >> act() >> F.batch_norm() >> F.dropout(0.2)
      d4 = conv1d(sig_len/8) >> act() >> F.batch_norm() >> F.dropout(0.2)
      d5 = conv1d(sig_len/16) >> act() >> F.batch_norm() >> F.dropout(0.2)
      d6 = conv1d(1)

      y = d1 >> d2 >> d3 >> d4 >> d5 >> d6
      return M.Model([x], [y(x)])


# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------

model, model2, encoder = make_model_2d(in_frames[0], n_latent, simple=False)
#model, model2, encoder = make_model_2d_conv2latent(in_frames[0])
#model, model2, encoder = make_model_2d_conv(in_frames[0], n_latent)

#model.load_weights('rope_texture2074_full_image___tae.hdf5')

#tools.print_layer_outputs(model)
model.summary()
loss_recorder = tools.LossRecorder()

#model2.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#tools.train(model2, in_frames, out_frames, 32, n_epochs//20, loss_recorder)
model.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
model2.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
#tools.train(model2, in_frames, out_frames, 128, n_epochs, loss_recorder)
tools.train(model, in_frames, out_frames[0], 32, n_epochs, loss_recorder)
#tools.train(model, in_frames, out_frames[0], 64, n_epochs, loss_recorder)
#tools.train(model, in_frames, out_frames[0], 64, 5*n_epochs, loss_recorder)
#tools.train(model, in_frames, out_frames[0], 128, 5*n_epochs, loss_recorder)

#model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#tools.train(model, in_frames, out_frames[0], 32, n_epochs//20, loss_recorder)
#model.compile(optimizer=keras.optimizers.Adam(), loss=loss_function)
#tools.train(model, in_frames, out_frames[0], 32, n_epochs, loss_recorder)


arnn_model = None
# arnn_model = make_model_2d_arnn(in_frames[0], simple=False)
# arnn_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_absolute_error)
# arnn_loss_recorder = tools.LossRecorder()
# tools.train(arnn_model, in_frames, next_samples, 32, n_epochs, arnn_loss_recorder)
# tools.train(arnn_model, in_frames, next_samples, 64, 5*n_epochs, arnn_loss_recorder)
# tools.train(arnn_model, in_frames, next_samples, 128, 5*n_epochs, arnn_loss_recorder)


# TODO write images to file

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


# ------------------------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------------------------

def plot_prediction_im(n=2000, signal_gen=make_signal, ofs=0):
   sig = signal_gen(n+100+ofs)[ofs:].T
   pred_sig = predict_signal(model, sig[:,:frame_size], shift, n+100)
   fig, ax = pl.subplots(5,1)
   ax[0].imshow(log(.1 + abs(sig[:n])), aspect='auto', cmap='gray')
   ax[1].imshow(log(.1 + abs(pred_sig[:n])), aspect='auto', cmap='gray')
   ax[2].imshow(log(.1 + abs(sig[:,:n]-pred_sig[:,:n])), aspect='auto', cmap='gray')

   if arnn_model:
      pred_sig_arnn = predict_ar_model(arnn_model, sig[:,:frame_size], n+100)
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



def plot_comp(model=model, n=2000, signal_gen=make_signal, ofs=0):
   sig = signal_gen(n+100+ofs)[ofs:].T
   pred_sig = predict_signal(model, sig[:,:frame_size], shift, n+100)
   fig, ax = pl.subplots(2,1, figsize=(8,3))
   ax[0].imshow(abs(sig[:n]), aspect='auto', cmap='gray')
   ax[0].set_xticks([])
   ax[1].imshow(abs(pred_sig[:n]), aspect='auto', cmap='gray')
   ax[1].plot([frame_size, frame_size], [0, n_nodes], '--r')
   ax[1].set_ylim([n_nodes-1, 0])
   ax[1].set_xlabel(r'$n$ (samples)', fontsize=16)
   pl.tight_layout()


def plot_image(model, n=10000, width=512):
   fig = pl.figure(figsize=(8,4))
   ax = pl.gca()
   img = predict_signal(model, np.random.rand(width, frame_size), shift, n)
   imshow(img, cmap='gray', aspect='auto')
   ax.plot([frame_size, frame_size], [0, width], '--r')
   ax.set_xlabel(r'$n$ (samples)', fontsize=16)
   ax.set_ylabel(r'$k$ (observable state)', fontsize=16)
   ax.set_xlim([0, n])
   ax.set_ylim([width, 0])
   # ax.set_yticks([-1,0,1])
   pl.tight_layout()
   #pl.colorbar()



#plot_prediction_im(1500, make_signal)
plot_comp(model, 1500, make_signal, 300)
