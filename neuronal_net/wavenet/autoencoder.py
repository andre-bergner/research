import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as K

import sys
sys.path.append('../')

from keras_tools import tools
from keras_tools import upsampling as Up
from keras_tools import functional as fun

# TODO
# • core-encoding layer must be width=height=1 --> stronger downsample 
# • core-encoding layer must be full connected --> bigger kernel
# • print size of core-encoder
# • TODO add additional loss: L2 of derivative


image_size = 16
num_feat0 = 16

#act = lambda: L.LeakyReLU(alpha=0.3)
act = lambda: L.Activation('tanh')
enc_activation = None # 'tanh'
dec_activation = None # 'tanh'
use_bias = False

#init = keras.initializers.uniform(1e-5,.01)
init = keras.initializers.VarianceScaling()
up_fill = 'repeat'
learning_rate =.1
loss_function = 'mean_absolute_error'
# loss_function = 'categorical_crossentropy'


x = L.Input(shape=(image_size, image_size,))
reshape_in = L.Reshape((image_size,image_size,1))
enc1 = L.Conv2D(num_feat0,   (5,5), padding='same', strides=(2,2), activation=enc_activation, kernel_initializer=init, use_bias=use_bias)
enc2 = L.Conv2D(num_feat0*2, (5,5), padding='same', strides=(2,2), activation=enc_activation, kernel_initializer=init, use_bias=use_bias)
enc3 = L.Conv2D(num_feat0*4, (5,5), padding='same', strides=(2,2), activation=enc_activation, kernel_initializer=init, use_bias=use_bias)
# enc_core = L.Conv2D(num_feat0*2, (1,1), activation=enc_activation, kernel_initializer=init)
enc_core = L.Conv2D(num_feat0, (2,2), padding='valid', activation=enc_activation, kernel_initializer=init)
dec1_up = L.UpSampling2D((4,4), fill=up_fill)
# dec1_up = L.UpSampling2D((2,2), fill=up_fill)
dec1    = L.Conv2D(num_feat0*2, (5,5), padding='same', activation=dec_activation, kernel_initializer=init, use_bias=use_bias)
dec2_up = L.UpSampling2D((2,2), fill=up_fill)
dec2    = L.Conv2D(num_feat0, (5,5), padding='same', activation=dec_activation, kernel_initializer=init, use_bias=use_bias)
dec3_up = L.UpSampling2D((2,2), fill=up_fill)
dec3    = L.Conv2D(1, (5,5), padding='same', activation=dec_activation, kernel_initializer=init, use_bias=use_bias)
reshape_out = L.Reshape((image_size,image_size))

deriv =  L.Conv2D(
   1, kernel_size=(2,2), padding='valid', strides=(2,2), use_bias=False,
   weights=[np.array([[[1,-1],[1,-1]]]).T], trainable=False)


# ring1_enc = fun.args >> reshape_in >> enc1 >> act()
# ring1_dec_ = fun.args >> dec3_up >> dec3 >> act()
# ring1_dec = ring1_dec_ >> reshape_out
# dy3 = y3 >> deriv >> ring1_dec

ring1_enc = fun.args >> reshape_in >> enc1 >> act()
ring1_dec = fun.args >> dec3_up >> dec3 >> act() >> reshape_out
y1 = ring1_enc >> ring1_dec
y2 = ring1_enc >> enc2 >> act() >> dec2_up >> dec2 >> act() >> ring1_dec
y3 = ring1_enc >> enc2 >> act() >> enc3 >> act() >> enc_core >> act() >> dec1_up >> dec1 >> act() >> dec2_up >> dec2 >> act() >> ring1_dec

model = M.Model([x], [y3(x)])
multi_model = M.Model([x], [y1(x),y2(x),y3(x)])
#deriv_model = M.Model([x], [dy3(x)])

# imshow(ss.convolve2d(images[0], [[1,-1],[1,-1]]))

model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)
multi_model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)

#model.load_weights('circle_auto_encoder.hdf5')


#code_input = L.Reshape((image_size,image_size,num_feat0*2), input_shape=(image_size,image_size,)))
#K.function([])


def print_layer_inputs(model):
   for l in model.layers:
      print(l.output_shape[1:])

def circle_img(x0, y0, r):
   xs = np.arange(image_size) - x0
   ys = np.arange(image_size) - y0
   img = np.array([r*r - (x*x + y*y) for x in xs for y in ys], dtype=float).reshape(image_size,image_size) / (2*r)
   return img * (img>=0)

def random_img(num_circles=4):
   img = np.zeros((image_size,image_size))
   rnd = np.random.rand
   for n in range(num_circles):
      img += circle_img(image_size*(0.8*rnd()+0.1), image_size*(0.8*rnd()+0.1), 0.2*image_size*rnd())
   return img * (img<=1) + (img>=1)

print('model shape')
print_layer_inputs(model)

print('generating test data')
images = np.array([random_img() for n in range(500)])
images_test = np.array([random_img() for n in range(200)])

print('training')
loss_recorder = tools.LossRecorder()
#tools.train(model, images, images, 20, 35, loss_recorder)
tools.train(multi_model, images, [images,images,images], 20, 35, loss_recorder)



import pylab as pl

def plot_orig_vs_reconst(n=0,data=images):
   fig, ax = pl.subplots(1,2)
   ax[0].imshow(data[n], cmap='gray')
   ax[1].imshow(model.predict(data[n:n+1])[0][:,:], cmap='gray')


def plot_kernels():
   fig, axes = pl.subplots(4,4)
   for ax,n in zip(flatten(axes), range(16)):
      ax.imshow(model.get_weights()[0][:,:,0,n], cmap='gray')

plot_orig_vs_reconst(0)


#from keras.utils import plot_model
#plot_model(model, to_file='wavelet_autoencoder.png')

