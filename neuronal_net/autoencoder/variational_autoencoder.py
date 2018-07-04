import sys
sys.path.append('../')

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.utils import plot_model

from keras_tools import functional_layers as F
from keras_tools import tools


width, height = 28, 28
x_dim = width * height
z_dim = 20
act = lambda a: L.Activation(a)

x = L.Input([width, height], name='x')
encoder = F.flatten() >> F.dense([160]) >> act('relu')
ex = encoder(x)


def sampling(args):
    z_mu, z_log_sigma = args
    batch = K.shape(z_mu)[0]
    dim = K.int_shape(z_mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mu + K.exp(0.5 * z_log_sigma) * epsilon

z_mu = F.dense([z_dim], name='z_mu')(ex)
z_log_sigma = F.dense([z_dim], name='z_log_sigma')(ex)
batch_size = K.shape(z_mu)[0]
epsilon = K.random_normal(shape=(batch_size, z_dim))
#z = z_mu + K.exp(0.5 * z_log_sigma) * epsilon
z = L.Lambda(sampling, output_shape=(z_dim,), name='z')([z_mu, z_log_sigma])
#exp_s = L.Lambda(lambda x: K.exp(0.5 * x), output_shape=(z_dim,))(z_log_sigma)
#z = L.Multiply([ exps(z_log_sigma), ])

decoder = (  F.dense([160]) >> act('relu')
          >> F.dense([width, height]) >> act('sigmoid')
          )

model = M.Model([x], [decoder(z)])

def vae_loss(y_true, y_pred):
   #reconst_loss = x_dim * keras.losses.mean_squared_error(y_true, y_pred)
   reconst_loss = K.mean(K.square(y_true-y_pred), axis=-1)
   kl_div = -.5 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma))
   return x_dim*reconst_loss + z_dim*kl_div

model.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss)

# plot_model(model, to_file='vae.png', show_shapes=True)

from keras.datasets import mnist
(data, _), (_, _) = mnist.load_data()
data = data/255.

loss_recorder = tools.LossRecorder()
tools.train(model, data, data, 32, 5, loss_recorder)



from pylab import *

def ims(n):
   img = model.predict(data[n:n+1])[0]
   fig, axs = subplots(1,2, figsize=(4,2))
   axs[0].imshow(data[n], cmap='gray')
   axs[1].imshow(img, cmap='gray')

ims(1)
ims(2)
ims(3)
