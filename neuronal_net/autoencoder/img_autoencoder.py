# TODO
# • try shared layers (like in the wavelet network)

from gen_autoencoder import *
from pylab import imshow

image_size = 32
n_features = 8
activation = None # 'tanh'

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = False
n_epochs = 5#30

#init = keras.initializers.uniform(1e-5,.01)
init = keras.initializers.VarianceScaling()


learning_rate =.1
loss_function = 'mean_absolute_error'
# loss_function = 'categorical_crossentropy'

model, joint_model = make_autoencoder([image_size,image_size], n_features)

model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)
joint_model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)



def print_layer_outputs(model):
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
print_layer_outputs(model)

print('generating test data')
images = np.array([random_img() for n in range(500)])
images_test = np.array([random_img() for n in range(200)])


print('training')
loss_recorder = tools.LossRecorder()
#tools.train(model, images, images, 20, n_epochs, loss_recorder)
tools.train(joint_model, images, [images,images,images], 20, n_epochs, loss_recorder)



import pylab as pl

def no_ticks(ax):
   ax.xaxis.set_visible(False)
   ax.yaxis.set_visible(False)
   return ax


def plot_orig_vs_reconst(n=0, data=images):
   fig, ax = pl.subplots(1,2)
   ax[0].imshow(data[n], cmap='gray')
   ax[1].imshow(model.predict(data[n:n+1])[0][:,:], cmap='gray')

def plot_orig_vs_reconst_n(ns, data=images):
   fig, ax = pl.subplots(len(ns),2)
   for k,n in enumerate(ns):
      no_ticks(ax[k,0]).imshow(data[n], cmap='gray')
      no_ticks(ax[k,1]).imshow(model.predict(data[n:n+1])[0][:,:], cmap='gray')


def plot_kernels():
   fig, axes = pl.subplots(4,4)
   for ax,n in zip(flatten(axes), range(16)):
      ax.imshow(model.get_weights()[0][:,:,0,n], cmap='gray')

#plot_orig_vs_reconst(0)
plot_orig_vs_reconst_n([1,2,3,4])

