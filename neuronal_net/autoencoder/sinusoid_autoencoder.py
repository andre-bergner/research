# TODO
# • create conv auto-encoder using huge kernel and stride to simulate dense layer → make slow transition
# • contractive AE
# • check distribution of freq, dist of samples-values
# • supervised AE: train sinusoid to map on circle/torus
# • do a PCA on sinusoids space to find guess for embedding dimension. Is it 2?
# • joint or interleaved greedy traing
# • plot distance matrix for sinusoids
#   -->  check distance matrix for codes of different sizes
#   -->  use as constrain?!
# • stacked AE vs greedy pre-training?
# • generate freq dist --> gen. sinusoids --> test if issues are due to dist or min/max freq
# • test several combinations of freq-dist, noise-levels, number of samples
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
from keras_tools import extra_layers as XL
from pylab import *

num_data = 512
sig_len = 128#256
noise_level = 0#.01

learning_rate = .1
loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'
loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

#activation = 'tanh'
activation = fun.bind(XL.tanhx, alpha=0.1)
#activation = lambda: L.LeakyReLU(alpha=0.3)
#activation = lambda x: XL.soft_relu(x, 0.1)
#activation = lambda x: keras.activations.relu(x, 0.1)



def gen_lin_freq(N=512, min=0, max=2*pi, random=False):
   if random:
      freqs = min + np.random.rand(N) * (max-min)
   else:
      freqs = np.linspace(min,max,N)
   return sort(freqs)

def gen_log_freq(N=512, min=0.1, max=0.9*pi, random=False):
   return np.exp(gen_lin_freq(N, np.log(min), np.log(max), random))

def sinusoid(freq, mod=0):
   t = np.arange(0,sig_len)
   return 0.8 * np.sin(freq*t + mod*sin(0.1*t))

def gen_sinusoids(N=512, freq_gen=gen_log_freq, signal_gen=sinusoid):
   return np.array([signal_gen(f) for f in freq_gen(N)])

def random_sinusoid(num_modes=1, decay=0.02):
   sig = np.zeros((sig_len))
   rnd = np.random.rand
   for n in range(num_modes):
      phi = np.exp( 1.j * np.pi * rnd() )
      sig += np.real(phi * np.exp( (-decay*np.exp(rnd()) + 1.j*(np.exp(-2.8*rnd()))) * np.arange(0,sig_len)))
      #sig += np.real(phi * np.exp( (-decay*np.exp(rnd()) + 1.j*(-2.8*rnd())) * np.arange(0,sig_len)))
   return sig / (2*num_modes)

def gen_sinousoid_and_circle(N=512, mod=0):
   t = np.arange(0,sig_len)
   W = np.linspace(0,2*np.pi,N)
   src_data = np.array([np.sin(w*t + mod*sin(0.1*t)) for w in W])
   dest_data = np.array([[np.sin(w), np.cos(w)] for w in W])
   return src_data#, dest_data

#data = gen_sinousoid_and_circle(mod=0.5)
#data = np.array([random_sinusoid(decay=0) for n in range(num_data)])
data = gen_sinusoids(freq_gen=fun.bind(gen_log_freq, max=0.5*pi, random=True))
#data = gen_sinusoids(freq_gen=fun.bind(gen_log_freq, min=1, max=0.9*pi))



def dense(units, use_bias=True):
   return fun.ARGS >> L.Dense(units=int(units), activation=activation, use_bias=use_bias)

"""
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
"""

def make_dense_model():

   x1 = L.Input([sig_len])
   x2 = L.Input([sig_len/2])
   x3 = L.Input([sig_len/4])

   enc1 = dense(sig_len/2)
   enc2 = dense(sig_len/4)
   enc3 = dense(16)

   dec3 = dense(sig_len/4)
   dec2 = dense(sig_len/2)
   dec1 = dense(sig_len)

   # enc1.activity_regularizer = keras.regularizers.l2(l=0.01)
   # enc2.activity_regularizer = keras.regularizers.l2(l=0.01)
   # enc2b.activity_regularizer = keras.regularizers.l2(l=0.01)
   # enc3.activity_regularizer = keras.regularizers.l2(l=0.01)

   ##y = enc1 >> enc2 >> enc2b >> enc3 >> dec3 >> dec2b >> dec2 >> dec1
   #y = enc1 >> enc2 >> enc3 >> dec3 >> dec2 >> dec1
   #y2 = enc1 >> enc2 >> dec2 >> dec1
   #y1 = enc1 >> dec1
   #return M.Model([x], [y(x)]), M.Model([x], [y(x), y1(x), y2(x)])

   y = enc1 >> enc2 >> enc3 >> dec3 >> dec2 >> dec1
   y1 = enc1 >> dec1
   y2 = enc2 >> dec2
   y3 = enc3 >> dec3
   return (
      M.Model([x1], [y(x1)]),
      M.Model([x1], [y1(x1)]),
      M.Model([x1], [enc1(x1)]),
      M.Model([x2], [y2(x2)]),
      M.Model([x2], [enc2(x2)]),
      M.Model([x3], [y3(x3)]),
      M.Model([x1], [(enc1 >> enc2 >> enc3)(x1)]),
   )


#model, joint_model = make_dense_model()
model, layer1, enc1, layer2, enc2, layer3, encoder = make_dense_model()


def train(model, inputs, target, batch_size, n_epochs, loss_recorder):
   if type(model.output) == list:
      target = len(model.output) * [target]
   tools.train(model, inputs, target, batch_size, n_epochs, loss_recorder)


print('model shape')
tools.print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()      # make this global


def train_model(model, data, fast=False):
   model.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
   train(model, tools.add_noise(data, noise_level), data, 64, 3000, loss_recorder)

   if fast: return

   model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
   train(model, tools.add_noise(data, noise_level), data, 64, 2000, loss_recorder)
   train(model, tools.add_noise(data, noise_level), data, 128, 2000, loss_recorder)

   # Pretraining works better without this:
   model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
   train(model, tools.add_noise(data, noise_level), data, 64, 2000, loss_recorder)
   train(model, tools.add_noise(data, noise_level), data, 128, 2000, loss_recorder)


train_model(layer1, data, fast=True)
code1 = enc1.predict(data)
train_model(layer2, code1, fast=True)
code2 = enc2.predict(code1)
train_model(layer3, code2, fast=True)
train_model(model, data)
code3 = encoder.predict(data)

# long term learning, fine tuning
# train(model, data, data, 512, 500000, loss_recorder)



figure()
semilogy(loss_recorder.losses)


plot_orig_vs_reconst = fun.bind(tools.plot_target_vs_prediction, model, data, data)
plot_top_and_worst = fun.bind(tools.plot_top_and_worst, model, data, data)

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((data[::step] - model.predict(data[::step])).T, 'k', alpha=0.2)

plot_top_and_worst()

#from keras.utils import plot_model
#plot_model(model, to_file='sig_autoencoder.png')


def dist(x,y):
   d = x - y
   return sqrt(np.dot(d,d))/len(x)

print("computing distances...")
distances0 = np.array([[dist(x,y) for x in data] for y in data])
distances1 = np.array([[dist(x,y) for x in code1] for y in code1])
distances2 = np.array([[dist(x,y) for x in code2] for y in code2])
distances3 = np.array([[dist(x,y) for x in code3] for y in code3])
