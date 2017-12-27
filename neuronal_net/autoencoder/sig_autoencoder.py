from gen_autoencoder import *
from pylab import imshow

sig_len = 256
n_features = 4
activation = None # 'tanh'

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = False
n_epochs = 30

#init = keras.initializers.uniform(1e-5,.01)
init = keras.initializers.VarianceScaling()


learning_rate =.1
loss_function = 'mean_absolute_error'
# loss_function = 'categorical_crossentropy'

model, joint_model = make_autoencoder([sig_len], n_features)

model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)
joint_model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)



def print_layer_outputs(model):
   for l in model.layers:
      print(l.output_shape[1:])

def random_img(num_modes=4):
   sig = np.zeros((sig_len))
   rnd = np.random.rand
   for n in range(num_modes):
      sig += np.real(np.exp( (-0.02*np.exp(rnd()) + 1.j*(np.exp(-2*rnd()))) * np.arange(0,sig_len)))
   return sig / (2*num_modes)


print('model shape')
print_layer_outputs(model)

print('generating test data')
images = np.array([random_img() for n in range(500)])

print('training')
loss_recorder = tools.LossRecorder()
#tools.train(model, images, images, 20, 30, loss_recorder)
tools.train(joint_model, images, [images,images,images], 20, n_epochs, loss_recorder)

import pylab as pl

def plot_orig_vs_reconst(n=0):
   fig = pl.figure()
   pl.plot(images[n])
   pl.plot(model.predict(images[n:n+1])[0])

plot_orig_vs_reconst(0)
