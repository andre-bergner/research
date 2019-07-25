from adversarial_ica import *

num_samples = 10000

layout = [2, 2]
w1 = np.pi*0.1
w2 = 0.2
sig1, sig2 = make_sin_gen(w1), make_sin_gen(w2)
sig_sum = (sig1 + sig2)(num_samples + 31)
signal = np.stack([ sig_sum[0:-4], sig_sum[1:-3], sig_sum[2:-2], sig_sum[3:-1] ])  # embedding
#signal = np.stack([
#   sig_sum[0:-16],
#   sig_sum[1:-15],
#   sig_sum[2:-14],
#   sig_sum[3:-13],
#   sig_sum[4:-12],
#   sig_sum[5:-11],
#   sig_sum[6:-10],
#   sig_sum[7:-9],
#   sig_sum[8:-8],
#   sig_sum[9:-7],
#   sig_sum[10:-6],
#   sig_sum[11:-5],
#   sig_sum[12:-4],
#   sig_sum[13:-3],
#   sig_sum[14:-2],
#   sig_sum[15:-1],
#])

#   analytical solution
mat = np.stack([
   np.cos( w1 * np.arange(4) ),
   np.sin( w1 * np.arange(4) ),
   np.cos( w2 * np.arange(4) ),
   np.sin( w2 * np.arange(4) ),
]).T
wat = np.linalg.inv(mat)
# plot( np.linalg.inv(M).dot(signal[:200]).T )


ica = AdversarialICA(
   signal,
   #discriminator_penalty=0.5,
   optimizer=keras.optimizers.Adam(lr=0.0001),
   critic_optimizer=keras.optimizers.Adam(lr=0.001),
   latent_space_layout=layout,
   instance_noise=.1
)

ica.train(500, batch_size=128, discriminator_runs=20)
plot_summary(ica, [0,2])
