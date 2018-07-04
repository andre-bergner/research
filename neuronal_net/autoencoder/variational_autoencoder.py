import sys
sys.path.append('../')

import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.utils import plot_model

from keras_tools import functional_layers as F
from keras_tools import extra_layers as XL
from keras_tools import tools



width, height = 28, 28
x_dim = width * height
z_dim = 20
act = lambda a: L.Activation(a)

x = L.Input([width, height], name='x')
encoder = F.flatten() >> F.dense([160]) >> act('relu') >> XL.VariationalEncoder(z_dim, x_dim)
z = encoder(x)

decoder = (  F.dense([160]) >> act('relu')
          >> F.dense([width, height]) >> act('sigmoid')
          )

model = M.Model([x], [decoder(z)])

model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

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
