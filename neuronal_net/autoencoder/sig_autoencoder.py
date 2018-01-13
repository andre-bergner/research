# TODO
# • create conv auto-encoder using huge kernel and stride to simulate dense layer → make slow transition
# • contractive AE
# • check distribution of freq, dist of samples-values
# • supervised AE: train sinusoid to map on circle/torus
# ✔

# OBSERVATIONS
# • logarithmicly distributed frequencies:
#   • low frequencies are learned very well
#   • the higher the frequency the harder to learn
# • linearly distributed frequencies:
#   • in general signal are learned harder
#   • all frequencies are learned equally good/bad
# • sinusoids are tori
#   • one sinusoid without phase (freq) lies on a circle, amplitude is radius
#   • one sinusoid with phase (freq & phase) lies on a torus (both wrap)
#   • two sinusoids without phase (freq1 & freq2) lie on a torus
#   • one decaying sinusoid without phase (freq & damp) lies on a spiral
#   • one decaying sinusoid with phase (freq & damp & phase) lies on donus-swiss-roll
#   • two decaying sinusoid without phase (freq1/2 & damp1/2) lies on ???
#     --> needs a fourth dimension, otherwise two entangled spirals would cut itself



from gen_autoencoder import *
from keras_tools import functional_layers as F
from pylab import imshow


def soft_relu(x, leak=0.0):
   return (0.5+leak) * x  +  (0.5-leak) * K.sqrt(1.0 + x*x)

keras.utils.generic_utils.get_custom_objects().update(
   {'soft_relu': L.Activation(soft_relu)}
)







num_data = 512
sig_len = 128#256

use_bias = False
n_features = 4
kern_len = 4

#act = lambda: L.LeakyReLU(alpha=0.3)
n_epochs = 30

learning_rate = .1
loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'
loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

#activation = 'tanh'
activation = lambda x: soft_relu(x,0.1)



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
   #to_latent = conv(16, 1, 2) >> act()
   #to_latent = conv(n_features, 1, 2) >> act()
   #from_latent = up(2) >> conv(16, 1) >> act()
   to_latent = fun.ARGS >> L.Flatten() >> F.dense([24]) >> act()
   from_latent = F.dense([16, 16]) >> act()

   dec4  = up(2) >> conv(8, 1) >> act()
   dec4b =          conv(8, kern_len) >> act()
   dec3  = up(2) >> conv(4, 1) >> act()
   dec3b =          conv(4, kern_len) >> act()
   dec2  = up(2) >> conv(2, 1) >> act()
   dec2b =          conv(2, kern_len) >> act()
   dec1  = up(2) >> conv(1, 1) >> act()
   dec1b =          conv(1, kern_len) >> act()

   # ----- debug -----------------------------------
   # m = M.Model([x], [(reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> to_latent >> from_latent)(x)])
   # for l in m.layers:
   #    print(l.output_shape[1:])

   # encoder = L.Dense(units=encoder_size, activation=activation, activity_regularizer=regularizers.l1(0.0001))#, weights=[np.eye(input_len), np.zeros(input_len)])
   # reshape2.activity_regularizer = keras.regularizers.l1(l=0.0001)


   y1 = reshape_in >> enc1 >> enc1b >> dec1 >> dec1b >> reshape_out
   y2 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y3 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y4 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> dec4 >> dec4b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y  = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> to_latent >> from_latent >> dec4 >> dec4b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out

   model = M.Model([x], [y(x)])
   joint_model = M.Model([x], [y1(x),y2(x),y3(x),y4(x),y(x)])

   return model, joint_model




def random_sinusoid(num_modes=1, decay=0.02):
   sig = np.zeros((sig_len))
   rnd = np.random.rand
   for n in range(num_modes):
      phi = np.exp( 1.j * np.pi * rnd() )
      sig += np.real(phi * np.exp( (-decay*np.exp(rnd()) + 1.j*(np.exp(-2.8*rnd()))) * np.arange(0,sig_len)))
      #sig += np.real(phi * np.exp( (-decay*np.exp(rnd()) + 1.j*(-2.8*rnd())) * np.arange(0,sig_len)))
   return sig / (2*num_modes)


print('generating test data')
data = np.array([random_sinusoid(decay=0) for n in range(num_data)])



def dense(units, use_bias=True):
   return fun.ARGS >> L.Dense(units=int(units), activation=activation, use_bias=use_bias)

def make_dense_model():
   x = input_like(data[0])
   #y = dense(sig_len/2) >> dense(4) >> dense(sig_len/2) >> dense(sig_len)
   enc1 = dense(sig_len/2)
   enc2 = dense(sig_len/4)
   #enc2b = dense(sig_len/4)
   enc3 = dense(16)
   dec3 = dense(sig_len/4)
   #dec2b = dense(sig_len/4)
   dec2 = dense(sig_len/2)
   dec1 = dense(sig_len)

   # enc1.activity_regularizer = keras.regularizers.l2(l=0.01)
   # enc2.activity_regularizer = keras.regularizers.l2(l=0.01)
   # enc2b.activity_regularizer = keras.regularizers.l2(l=0.01)
   # enc3.activity_regularizer = keras.regularizers.l2(l=0.01)

   #y = enc1 >> enc2 >> enc2b >> enc3 >> dec3 >> dec2b >> dec2 >> dec1
   y = enc1 >> enc2 >> enc3 >> dec3 >> dec2 >> dec1
   y2 = enc1 >> enc2 >> dec2 >> dec1
   y1 = enc1 >> dec1
   return M.Model([x], [y(x)]), M.Model([x], [y(x), y1(x), y2(x)])



#model, joint_model = make_model([sig_len], model_gen, use_bias=use_bias)
model, joint_model = make_dense_model()


def train(model, inputs, target, batch_size, n_epochs, loss_recorder):
   if type(model.output) == list:
      target = len(model.output) * [target]
   tools.train(model, inputs, target, batch_size, n_epochs, loss_recorder)


print('model shape')
tools.print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()



#joint_model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#train(joint_model, data, data, 32, 2000, loss_recorder)

def add_noise(data, sigma=0.0):
   return data + sigma * np.random.randn(*data.shape)

model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
train(model, add_noise(data), data, 64, 3000, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
train(model, add_noise(data), data, 64, 2000, loss_recorder)
train(model, add_noise(data), data, 128, 2000, loss_recorder)

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
train(model, add_noise(data), data, 64, 2000, loss_recorder)
train(model, add_noise(data), data, 128, 2000, loss_recorder)


# long term learning, fine tuning
# train(model, data, data, 512, 500000, loss_recorder)


# train(joint_model, data, data, 20, n_epochs, loss_recorder)
# train(model, data, data, 20, n_epochs, loss_recorder)


from pylab import *

figure()
semilogy(loss_recorder.losses)

#model.load_weights('circle_auto_encoder.hdf5')
#model.save_weights('sig_autoencoder_decently_trained.hdf5')


import pylab as pl

def plot_orig_vs_reconst(n=0, ax=None):
   if ax == None:
      pl.figure()
      plot = pl.plot
   else:
      plot = ax.plot
   plot(data[n])
   plot(model.predict(data[n:n+1])[0])

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((data[::step] - model.predict(data[::step])).T, 'k', alpha=0.2)

def plot_top_and_worst(num=3):
   dist = [(model.evaluate(data[n:n+1],data[n:n+1],verbose=0),n) for n in range(len(data))]
   dist.sort()
   fig, ax = pl.subplots(num,2)
   for n in range(num):
      plot_orig_vs_reconst(dist[n][1], ax[n,0])
      plot_orig_vs_reconst(dist[-n-1][1], ax[n,1])


plot_top_and_worst()

#from keras.utils import plot_model
#plot_model(model, to_file='sig_autoencoder.png')