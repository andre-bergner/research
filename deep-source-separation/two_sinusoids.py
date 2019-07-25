from separator import *

fixed_sin = make_sin_gen(np.pi*0.1)
scan_sin = make_sin_gen(0.1)
sig1, sig2 = fixed_sin, scan_sin

signal = sig1 + sig2

frame_size = 32
latent_sizes = [2, 2]

coder_factory=DenseFactory(
   input_size=frame_size,
   latent_sizes=latent_sizes,
   decoder_noise={"stddev": .5, "decay": 0.0001, "final_stddev": 0.05},
   #layer_sizes=[]    # no hidden layers, straight into latent space
)

def make_separator():
   return Separator(
      signal=signal(5000),
      coder_factory=coder_factory,
      signal_gens=[sig1, sig2],
      loss='mae',
      adversarial=0.5,
      critic_runs=10,
      optimizer=keras.optimizers.Adam(0.0001),  # .0001 better than .001
   )

# sep = make_separator()
# train_and_summary(sep, 10, batch_size=16)

### Run the model from scratch many times:
seps = []
for _ in range(10):
   sep = make_separator()
   seps.append(sep)
   train_and_summary(sep, 150)
