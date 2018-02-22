from imports import *

default_configuration = {
   "frame_size": 128,
   "shift": 8,
   "n_latent": 4,
   "in_noise_stddev": 0.05,
   "code_noise_stddev": 0.01,
}


def models(config=default_configuration):

   activation = fun.bind(XL.tanhx, alpha=0.1)

   act = lambda: L.Activation(activation)
   softmax = lambda: L.Activation(L.activations.softmax)
   eta1 = lambda: F.noise(config["in_noise_stddev"])
   eta2 = lambda: F.noise(config["code_noise_stddev"])

   n_latent = config["n_latent"]

   """
   def arnn_model(example_frame):
      x = F.input_like(example_frame)

      d1 = F.dense([int(frame_size/2)]) >> act()
      d2 = F.dense([int(frame_size/4)]) >> act()
      d3 = F.dense([n_latent]) >> act()
      d4 = F.dense([int(frame_size/2)]) >> act()
      d5 = F.dense([int(frame_size/4)]) >> act()
      d6 = F.dense([1]) >> act()

      #d1 = F.dense([int(frame_size/2)]) >> act()
      #d2 = F.dense([int(frame_size/4)]) >> act()
      #d3 = F.dense([int(frame_size/8)]) >> act()
      #d4 = F.dense([1]) >> act()
      # y = eta() >> d1 >> d2 >> d3 >> d4
      # return M.Model([x], [y(x)])
      chain = d1 >> d2 >> d3 >> eta2() >> d4 >> d5 >> d6
      y1 = (eta1() >> chain)(x)
      x2 = L.concatenate([XL.Slice(XL.SLICE_LIKE[:,1:])(x), y1])
      y2 = chain(x2)

      return M.Model([x], [y1]), M.Model([x], [y1, y2])
   """

   def arnn_model(example_frame):
      frame_size = example_frame.shape[-1]

      x = F.input_like(example_frame)

      d1 = F.dense([int(frame_size/2)]) >> act()
      d2 = F.dense([int(frame_size/4)]) >> act()
      d3 = F.dense([n_latent]) >> act()
      d4 = F.dense([int(frame_size/2)]) >> act()
      d5 = F.dense([int(frame_size/4)]) >> act()
      d6 = F.dense([1]) >> act()

      chain = eta1() >> d1 >> d2 >> d3 >> eta2() >> d4 >> d5 >> d6

      return M.Model([x], [chain(x)])


   def parnn_model(example_frame, bins=64):
      frame_size = example_frame.shape[-1]

      x = F.input_like(example_frame)

      d1 = F.dense([int(frame_size/2)]) >> act()
      d2 = F.dense([int(frame_size/2)]) >> act()
      d3 = F.dense([int(frame_size/4)]) >> act()
      d4 = F.dense([int(frame_size/4)]) >> act()
      d5 = F.dense([bins]) >> softmax()

      chain = eta1() >> d1 >> d2 >> d3 >> d4 >> d5

      return M.Model([x], [chain(x)])


   def tae_model2(example_frame):
      frame_size = example_frame.shape[-1]

      x = F.input_like(example_frame)

      enc1 = F.dense([int(frame_size/2)]) >> act()
      enc2 = F.dense([int(frame_size/4)]) >> act() >> F.dropout(0.4) # >> F.batch_norm() >> F.dropout(0.2)
      enc3 = F.dense([int(frame_size/8)]) >> act() >> F.dropout(0.4) # >> F.batch_norm() >> F.dropout(0.2)
      enc4 = F.dense([n_latent]) >> act()
      dec4 = F.dense([int(frame_size/8)]) >> act() >> F.dropout(0.4) #>> F.batch_norm() >> F.dropout(0.2)
      dec3 = F.dense([int(frame_size/2)]) >> act() >> F.dropout(0.4) #>> F.batch_norm() >> F.dropout(0.2)
      dec2 = F.dense([int(frame_size/4)]) >> act()
      dec1 = F.dense([frame_size]) #>> act()

      encoder = enc1 >> enc2 >> enc3 >> enc4
      decoder = dec4 >> dec3 >> dec2 >> dec1
      chain = eta1() >> encoder >> eta2() >> decoder
      latent = encoder(x)
      out = decoder(latent)

      return M.Model([x], [out]), M.Model([x], [out, chain(out)]), M.Model([x], [latent])#, XL.jacobian(latent,x)


   def tae_model(example_frame):
      frame_size = np.size(example_frame)
      x = F.input_like(example_frame)

      enc1 = F.dense([int(frame_size/2)]) >> act()
      enc2 = F.dense([int(frame_size/4)]) >> act()
      enc3 = F.dense([n_latent]) >> act()
      dec3 = F.dense([int(frame_size/2)]) >> act()
      dec2 = F.dense([int(frame_size/4)]) >> act()
      dec1 = F.dense([frame_size]) #>> act()

      encoder = enc1 >> enc2 >> enc3
      decoder = dec3 >> dec2 >> dec1
      chain = eta1() >> encoder >> eta2() >> decoder
      latent = encoder(x)
      out = decoder(latent)

      return M.Model([x], [out]), M.Model([x], [out, chain(out)]), M.Model([x], [latent]), XL.jacobian(latent,x)

   return arnn_model, parnn_model, tae_model, tae_model2


# transformations needed for the probabilistic model

def p2x(p):
   return np.inner(np.linspace(-1,1,len(p)), p)

def x2p(x, n=64, dev=.02):
   p = np.exp( -(np.linspace(-1,1,n)-x)**2 / dev**2 )
   return p / sum(p)
