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


image_size = 32
enc_activation = 'relu'
dec_activation = 'relu'
num_feat0 = 16

model = M.Sequential()

model.add(L.Reshape((image_size,image_size,1), input_shape=(image_size,image_size,)))
model.add(L.Conv2D(num_feat0, (5,5), padding='same', strides=(2,2), activation=enc_activation))
model.add(L.Conv2D(num_feat0*2, (5,5), padding='same', strides=(2,2), activation=enc_activation))
model.add(L.Conv2D(num_feat0*4, (5,5), padding='same', strides=(2,2), activation=enc_activation))

model.add(L.Conv2D(num_feat0*2, (1,1), activation=enc_activation))

model.add(L.UpSampling2D((2,2)))
model.add(L.Conv2D(num_feat0*2, (5,5), padding='same', activation=dec_activation))
model.add(L.UpSampling2D((2,2)))
model.add(L.Conv2D(num_feat0, (5,5), padding='same', activation=dec_activation))
model.add(L.UpSampling2D((2,2)))
model.add(L.Conv2D(1, (5,5), padding='same', activation=dec_activation))
model.add(L.Reshape((image_size,image_size)))

model.compile(optimizer=keras.optimizers.SGD(lr=.01), loss='mean_absolute_error')
#model.compile(optimizer=keras.optimizers.SGD(lr=.002), loss='categorical_crossentropy')
#model.compile(optimizer=keras.optimizers.SGD(lr=.001), loss='binary_crossentropy')

#model.load_weights('circle_auto_encoder.hdf5')


#code_input = L.Reshape((image_size,image_size,num_feat0*2), input_shape=(image_size,image_size,)))
#K.function([])


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

print('generating test data')
images = np.array([random_img() for n in range(500)])

print('training')
loss_recorder = tools.LossRecorder()
tools.train(model, images, images, 20, 10, loss_recorder)



import pylab as pl

def plot_orig_vs_reconst(n=0):
   fig, ax = pl.subplots(1,2)
   ax[0].imshow(images[n], cmap='gray')
   ax[1].imshow(model.predict(images[n:n+1])[0][:,:], cmap='gray')


def plot_kernels():
   fig, axes = pl.subplots(4,4)
   for ax,n in zip(flatten(axes), range(16)):
      ax.imshow(model.get_weights()[0][:,:,0,n], cmap='gray')



