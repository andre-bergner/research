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
from pylab import *
from keras_tools import extra_layers as XL


num_data = 512
sig_len = 128#256

use_bias = True

n_epochs = 30

learning_rate = .1
loss_function = 'mean_absolute_error'
# loss_function = 'mean_squared_error'
# loss_function = 'categorical_crossentropy'
loss_function = lambda y_true, y_pred: \
   keras.losses.mean_squared_error(y_true, y_pred) + keras.losses.mean_absolute_error(y_true, y_pred)

#activation = 'tanh'
#activation = lambda x: Xl.soft_relu(x,0.1)
#act = lambda: L.LeakyReLU(alpha=0.3)
activation = fun.bind(XL.tanhx, alpha=0.1)



def gen_sinousoid_and_circle(N=512):
   t = np.arange(0,sig_len)
   W = np.linspace(0,2*np.pi,N)
   src_data = np.array([np.sin(w*t) for w in W])
   dest_data = np.array([[np.sin(w), np.cos(w)] for w in W])
   #dest_data = np.array([[np.cos(w), np.sin(w)] for w in W])
   return src_data, dest_data

src_data, dest_data = gen_sinousoid_and_circle()
# TODO name feature, code



def dense(units, use_bias=True):
   return fun.ARGS >> L.Dense(units=int(units), activation=activation, use_bias=use_bias)

def make_encoder_model():
   x = input_like(src_data[0])
   enc1 = dense(sig_len/2)
   enc2 = dense(sig_len/4)
   enc3 = dense(len(dest_data[0]))
   y = enc1 >> enc2 >> enc3
   return M.Model([x], [y(x)])

def make_decoder_model():
   con = L.concatenate
   add = L.add
   x = input_like(dest_data[0])

   #d1 = con([x,dense(2)(x)])
   #d2 = con([x,d1,dense(4)(d1)])
   #d3 = con([x,d1,d2,dense(8)(d2)])
   #d4 = con([x,d1,d2,d3,dense(16)(d3)])
   #d5 = con([x,d1,d2,d3,d4,dense(32)(d3)])

   #d1 = dense(4)(x)
   #d2 = dense(8)(d1)
   #d3 = dense(8)(d2)
   #d4 = dense(8)(d3)
   #d5 = dense(8)(d4)
   #d6 = dense(8)(d5)
   #dc = con([x,d1,d2,d3,d4,d5,d6])
   #y = (dense(sig_len) >> dense(sig_len))(dc)
   ##y = (dense(sig_len) >> dense(sig_len) >> dense(sig_len))(dc)

   d1a = dense(4)(x)
   d1b = dense(4)(d1a)
   r1 = add([d1a,d1b])
   d2a = dense(8)(r1)
   d2b = dense(8)(d1a)
   r2 = add([d2a,d2b])
   d3a = dense(16)(r2)
   d3b = dense(16)(d3a)
   r3 = add([d3a,d3b])
   c1 = con([x,r1,r2,r3])
   y = (dense(sig_len) >> dense(sig_len))(c1)

   return M.Model([x], [y])


#model, joint_model = make_model([sig_len], model_gen, use_bias=use_bias)
#model, joint_model = make_dense_model()
encoder = make_encoder_model()
decoder = make_decoder_model()


def train(model, inputs, target, batch_size, n_epochs, loss_recorder):
   if type(model.output) == list:
      target = len(model.output) * [target]
   tools.train(model, inputs, target, batch_size, n_epochs, loss_recorder)


print('model shape')
#tools.print_layer_outputs(model)

print('training')
loss_recorder = tools.LossRecorder()



#encoder.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
#train(encoder, src_data, dest_data, 64, 3000, loss_recorder)
#
#encoder.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
#train(encoder, src_data, dest_data, 64, 2000, loss_recorder)
#train(encoder, src_data, dest_data, 128, 2000, loss_recorder)
#
#encoder.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
#train(encoder, src_data, dest_data, 64, 2000, loss_recorder)
#train(encoder, src_data, dest_data, 128, 2000, loss_recorder)


#figure()
#semilogy(loss_recorder.losses)
#
#figure()
#plot(encoder.predict(src_data) - dest_data)


decoder.compile(optimizer=keras.optimizers.SGD(lr=0.5), loss=loss_function)
train(decoder, dest_data, src_data, 64, 10000, loss_recorder)

decoder.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss=loss_function)
train(decoder, dest_data, src_data, 64, 2000, loss_recorder)
train(decoder, dest_data, src_data, 128, 2000, loss_recorder)
train(decoder, dest_data, src_data, 256, 2000, loss_recorder)

decoder.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss=loss_function)
train(decoder, dest_data, src_data, 64, 2000, loss_recorder)
train(decoder, dest_data, src_data, 128, 2000, loss_recorder)


figure()
semilogy(loss_recorder.losses)

#figure()
#plot(decoder.predict(dest_data) - src_data, 'k')

plot_orig_vs_reconst = fun.bind(tools.plot_target_vs_prediction, decoder, dest_data, src_data)

plot_orig_vs_reconst(24)
