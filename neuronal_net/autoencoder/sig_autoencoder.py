from gen_autoencoder import *
from pylab import imshow

sig_len = 256
n_features = 4
kern_len = 16

#act = lambda: L.LeakyReLU(alpha=0.3)
use_bias = False
n_epochs = 30

learning_rate = .1
loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'

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
   to_latent = conv(n_features, 1, 2) >> act()
   #to_latent = conv(16, 1, 2) >> act()
   from_latent = up(2) >> conv(16, 1) >> act()

   dec4  = up(2) >> conv(8, 1) >> act()
   dec4b =          conv(8, kern_len) >> act()
   dec3  = up(2) >> conv(4, 1) >> act()
   dec3b =          conv(4, kern_len) >> act()
   dec2  = up(2) >> conv(2, 1) >> act()
   dec2b =          conv(2, kern_len) >> act()
   dec1  = up(2) >> conv(1, 1) >> act()
   dec1b =          conv(1, kern_len) >> act()

   # ----- debug -----------------------------------
   # m = M.Model([x], [(reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc4 >> to_latent >> dec4 >> dec3 >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out)(x)])
   # for l in m.layers:
   #    print(l.output_shape[1:])

   y1 = reshape_in >> enc1 >> enc1b >> dec1 >> dec1b >> reshape_out
   y2 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y3 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y4 = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> dec4 >> dec4b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out
   y  = reshape_in >> enc1 >> enc1b >> enc2 >> enc2b >> enc3 >> enc3b >> enc4 >> enc4b >> to_latent >> from_latent >> dec4 >> dec4b >> dec3 >> dec3b >> dec2 >> dec2b >> dec1 >> dec1b >> reshape_out

   model = M.Model([x], [y(x)])
   joint_model = M.Model([x], [y1(x),y2(x),y3(x),y4(x),y(x)])

   return model, joint_model


model, joint_model = make_model([sig_len], model_gen, use_bias=use_bias)

model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)
joint_model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss=loss_function)





def print_layer_outputs(model):
   for l in model.layers:
      print(l.output_shape[1:])

def random_img(num_modes=1):
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
#tools.train(model, images, images, 20, n_epochs, loss_recorder)
#tools.train(joint_model, images, [images,images,images,images,images], 20, n_epochs, loss_recorder)


tools.train(joint_model, images, [images,images,images,images,images], 20, 200, loss_recorder)

joint_model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
tools.train(joint_model, images, [images,images,images,images,images], 50, 200, loss_recorder)

joint_model.compile(optimizer=keras.optimizers.SGD(lr=0.001), loss=loss_function)
tools.train(joint_model, images, [images,images,images,images,images], 100, 200, loss_recorder)


from pylab import *

figure()
semilogy(loss_recorder.losses)

figure()

#model.load_weights('circle_auto_encoder.hdf5')
#model.save_weights('sig_autoencoder_decently_trained.hdf5')


import pylab as pl

def plot_orig_vs_reconst(n=0):
   fig = pl.figure()
   pl.plot(images[n])
   pl.plot(model.predict(images[n:n+1])[0])

def plot_diff(step=10):
   fig = pl.figure()
   pl.plot((images[::step] - model.predict(images[::step])).T, 'k', alpha=0.2)

plot_orig_vs_reconst(0)
