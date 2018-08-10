from imports import *
from coder_factories import *
from pylab import *

factor = 1
frame_size = factor*128
n_pairs = 10000
latent_sizes = [2, 2]
n_epochs = 10
noise_stddev = 0.0#5


fixed_sin = make_sin_gen(np.pi*0.1)
scan_sin = make_sin_gen(0.1)
sig1, sig2 = fixed_sin, scan_sin
#sig1, sig2 = two_sin
sig_gen = sig1 + sig2

frames = np.array([w for w in windowed(sig_gen(n_pairs+frame_size), frame_size, 1)])

#factory = DenseFactory(frames[0], latent_sizes)
factory = ConvFactory(frames[0], latent_sizes, kernel_size=3, use_batch_norm=False, scale_factor=factor)
model, encoder, model_sf, [mode1, mode2] = make_factor_model(
   frames[0], factory, noise_stddev=noise_stddev, shared_encoder=True)
model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001), loss='mse')
model.load_weights('init_fail.hdf5')
#model.save_weights('init_weights.hdf5')

loss_recorder = LossRecorder(lambda n,m: pred_error([mode1,mode2][n], frames, [sig1,sig2][m], 2048))

tools.train(model, frames, frames, 128, n_epochs, loss_recorder)

training_summary(model, mode1, mode2, encoder, sig_gen, sig1, sig2, frames, loss_recorder)

# TODO
# • analyze eigen values of layers → plot EV of successful vs unsuccessful trainings
#
# for _ in range(10):
#    loss_recorder = LossRecorder()
#    model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0001), loss='mse')
#    model.load_weights('init_fail.hdf5')
#    tools.train(model, frames, frames, 128, n_epochs, loss_recorder)
#    training_summary(model, mode1, mode2, encoder, sig_gen, sig1, sig2, frames, loss_recorder)
