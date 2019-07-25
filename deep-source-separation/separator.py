from imports import *
from coder_factories import *
from keras_tools import test_signals as TS
from pylab import *



def train_batch(model, x_data, y_data, on_batch, batch_size, n_epochs, callbacks=[]):

   n_samples = x_data.shape[0]
   batches_per_epoch = n_samples // batch_size
   params = {"epochs": n_epochs, "samples": n_samples, "batch_size": batch_size}

   for c in callbacks:
      c.params = params
      c.model = model
      c.on_train_begin()

   for n_epoch in range(n_epochs):

      for c in callbacks:
         c.on_epoch_begin(n_epoch)

      for n_batch in range(batches_per_epoch):

         for c in callbacks:
            c.on_batch_begin(n_batch)

         loss = on_batch(batch_size, n_batch)

         for c in callbacks:
            c.on_batch_end(n_batch, {"loss": loss})
      
      for c in callbacks:
         c.on_epoch_end(n_epoch)

   for c in callbacks:
      c.on_train_end()



class Critic:

   def __init__(self, z_dim, optimizer, instance_noise=None):
      z_in = L.Input(shape=(z_dim,))

      # l = (F.dense([40]) >> L.LeakyReLU(0.1))(z_in)
      # l = L.concatenate([(F.dense([40]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([30]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([30]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([20]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = (F.dense([1]) >> L.Activation('sigmoid'))(l)

      # l = (F.dense([100]) >> L.LeakyReLU(0.1))(z_in)
      # l = L.concatenate([(F.dense([100]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([100]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([100]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([50]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([50]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([50]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = L.concatenate([(F.dense([20]) >> L.LeakyReLU(0.1))(l), z_in])
      # l = (F.dense([1]) >> L.Activation('sigmoid'))(l)

      l = (  F.dense([100]) >> L.LeakyReLU(0.1) 
          >> F.dense([100]) >> L.LeakyReLU(0.1) 
          >> F.dense([1]) >> L.Activation('sigmoid')
          )(z_in)

      self.model = M.Model([z_in], [l])

      self.train_model = M.Model([z_in], [l])
      #grad = XL.jacobian(l, z_in)
      #self.train_model.add_loss(K.mean(0.1*K.square(grad)))
      self.train_model.compile(optimizer=optimizer, loss='binary_crossentropy')

      self.instance_noise = instance_noise

   @staticmethod
   def fake_labels(n):
      return np.zeros((n, 1))

   @staticmethod
   def real_labels(n):
      return np.ones((n, 1))

   def train(self, separator, batch_size):
      n_samples = separator.frames.shape[0]
      batch_half = batch_size//2
      rnd_idx = lambda: np.random.randint(0, n_samples, batch_half)

      z_depend = separator.encoder.predict(separator.frames[rnd_idx()])
      z_factor = separator.encoder.predict(separator.frames[rnd_idx()])
      z_factor2 = separator.encoder.predict(separator.frames[rnd_idx()])
      z_factor[:,:separator.latent_sizes[0]] = z_factor2[:,:separator.latent_sizes[0]]

      batch = np.concatenate([z_factor, z_depend])
      if self.instance_noise:
         batch += self.instance_noise * np.random.randn(*batch.shape)
      label = np.concatenate([self.real_labels(batch_half), self.fake_labels(batch_half)])
      return self.train_model.train_on_batch(batch, label)


class WassersteinCritic:

   def __init__(self, z_dim, optimizer):
      z_in = L.Input(shape=(z_dim,))

      l = (F.dense([100]) >> L.LeakyReLU(0.1))(z_in)
      l = L.concatenate([(F.dense([100]) >> L.LeakyReLU(0.1))(l), z_in])
      l = L.concatenate([(F.dense([100]) >> L.LeakyReLU(0.1))(l), z_in])
      l = L.concatenate([(F.dense([100]) >> L.LeakyReLU(0.1))(l), z_in])
      l = L.concatenate([(F.dense([50]) >> L.LeakyReLU(0.1))(l), z_in])
      l = L.concatenate([(F.dense([50]) >> L.LeakyReLU(0.1))(l), z_in])
      l = L.concatenate([(F.dense([50]) >> L.LeakyReLU(0.1))(l), z_in])
      l = L.concatenate([(F.dense([20]) >> L.LeakyReLU(0.1))(l), z_in])
      l = F.dense([1])(l)

      wasserstein_loss = lambda y_true, y_pred: K.mean(y_true * y_pred)

      self.model = M.Model([z_in], [l])
      self.model.compile(optimizer=optimizer, loss=wasserstein_loss)

      self.clip_value = 0.01

   @staticmethod
   def fake_labels(n):
      return np.ones((n, 1))

   @staticmethod
   def real_labels(n):
      return -np.ones((n, 1))

   def train(self, separator, batch_size):
      n_samples = separator.frames.shape[0]
      batch_half = batch_size//2
      rnd_idx = lambda: np.random.randint(0, n_samples, batch_half)

      z_depend = separator.encoder.predict(separator.frames[rnd_idx()])
      z_factor = separator.encoder.predict(separator.frames[rnd_idx()])
      z_factor2 = separator.encoder.predict(separator.frames[rnd_idx()])
      z_factor[:,:separator.latent_sizes[0]] = z_factor2[:,:separator.latent_sizes[0]]

      batch = np.concatenate([z_factor, z_depend])
      label = np.concatenate([self.real_labels(batch_half), self.fake_labels(batch_half)])
      loss = self.model.train_on_batch(batch, label)

      c = self.clip_value
      for l in self.model.layers:
         l.set_weights([np.clip(w, -c, c) for w in l.get_weights()])

      return np.exp(loss)



class TimeDomainCritic:
   """
   This critic is applied in the final layers of the whole encoder stack, i.e. on the
   layers before the final summing operation.
   """

   def __init__(self, y_dim, optimizer):
      y_in = L.Input(shape=(y_dim, 2))
      flat_y = L.Flatten()(y_in)

      l = (F.dense([y_dim]) >> L.LeakyReLU(0.1))(flat_y)
      l = L.concatenate([(F.dense([y_dim]) >> L.LeakyReLU(0.1))(l), flat_y])
      l = L.concatenate([(F.dense([y_dim//2]) >> L.LeakyReLU(0.1))(l), flat_y])
      l = L.concatenate([(F.dense([y_dim//4]) >> L.LeakyReLU(0.1))(l), flat_y])
      l = L.concatenate([(F.dense([y_dim//8]) >> L.LeakyReLU(0.1))(l), flat_y])
      l = (F.dense([1]) >> L.Activation('sigmoid'))(l)

      self.model = M.Model([y_in], [l])

      self.train_model = M.Model([y_in], [l])
      self.train_model.compile(optimizer=optimizer, loss='binary_crossentropy')


   @staticmethod
   def fake_labels(n):
      return np.zeros((n, 1))

   @staticmethod
   def real_labels(n):
      return np.ones((n, 1))

   def train(self, separator, batch_size):
      n_samples = separator.frames.shape[0]
      idx = np.random.randint(0, n_samples, batch_size)
      y0 = separator.modes[0].predict(separator.frames[idx])
      np.random.shuffle(idx[:batch_size//2])
      y1 = separator.modes[1].predict(separator.frames[idx])

      batch = np.stack([y0, y1], axis=-1)
      label = np.concatenate([np.ones((batch_size//2, 1)), np.zeros((batch_size//2, 1))])
      return self.train_model.train_on_batch(batch, label)





class Separator:

   def __init__(self
   ,  signal
   ,  coder_factory
   ,  signal_gens
   ,  input_noise=None
   ,  latent_noise=None
   ,  loss='mse'
   ,  adversarial=None
   ,  critic_runs=None
   ,  optimizer=keras.optimizers.Adam(0.0001, 0.7)
   ,  critic_optimizer=None
   ,  verbose=False
   ,  vanishing_xprojection=False
   ,  info_loss=None
   ,  custom_features=None
   ,  custom_noises=None
   ,  stride=1
   ):
      frame_size = coder_factory.input_size
      assert(type(frame_size) == int)

      self.frames = np.array([w[::stride] for w in windowed(signal, stride*frame_size, 1)])
      self.stride = stride

      self.model, self.encoder, _, self.modes = make_factor_model(
         self.frames[0], coder_factory,
         input_noise=input_noise,
         shared_encoder=True,
         vanishing_xprojection=vanishing_xprojection,
         info_loss=info_loss,
         latent_noise=latent_noise,
         custom_features=custom_features,
         custom_noises=custom_noises
      )

      self.model.compile(optimizer=optimizer, loss=loss)


      def infer(model, num=None):
         return build_prediction(model, self.frames, num=num, stride=stride)

      self.model.infer = infer.__get__(self.model)
      self.modes[0].infer = infer.__get__(self.modes[0])  # TODO loop
      self.modes[1].infer = infer.__get__(self.modes[1])


      # ________________________________________________________________________
      # CRITIC

      if adversarial and not critic_runs: critic_runs = 10

      if critic_runs:

         if not critic_optimizer:
            critic_optimizer = optimizer

         self.latent_sizes = coder_factory.latent_sizes
         #self.critic = Critic(sum(self.latent_sizes), optimizer, instance_noise=0.1)
         self.critic = WassersteinCritic(sum(self.latent_sizes), optimizer)
         #self.critic = TimeDomainCritic(coder_factory.input_size, optimizer)

         if adversarial:

            self.critic.model.trainable = False

            self.combined = M.Model(
               [self.model.input],
               [self.model.output, self.critic.model(self.encoder.output)]
               # [self.model.output, self.critic.model(XL.stack([m.output for m in self.modes]))]
            )
            # self.combined.add_loss(0.005*K.square(K.mean(self.encoder.outputs[0])))
            # self.combined.add_loss(0.005*K.square(1-K.mean(K.square(self.encoder.outputs[0]))))
            self.combined.compile(
               loss=[loss, 'binary_crossentropy'],
               loss_weights=[1. - adversarial, adversarial],
               optimizer=optimizer
            )


      def on_batch(batch_size, n_batch):
         n_samples = self.frames.shape[0]
         idx = np.random.randint(0, n_samples, batch_size)
         batch = self.frames[idx]

         if adversarial:
            dis_w1 = self.critic.model.get_weights()[0].copy()
            _, loss, _ = self.combined.train_on_batch([batch], [batch, self.critic.real_labels(batch_size)])
            assert( (dis_w1 == self.critic.model.get_weights()[0]).all() )
         else:
            loss = self.model.train_on_batch(batch, batch)

         if critic_runs:
            for _ in range(critic_runs):
               d_loss = self.critic.train(self, batch_size)
            return np.array([loss, d_loss])

         return np.array([loss])

      self.on_batch = on_batch


      if verbose:
         self.model.summary()
         plot_model(self.model, to_file='ssae.png', show_shapes=True)



      class SeparationRecorder(keras.callbacks.Callback):

         def __init__(self, modes, src_gens, num_samples=5000, **kargs):
            super(SeparationRecorder, self).__init__(**kargs)
            self.mutual_information = []
            self.pred_errors = []
            self.modes = modes
            self.src = [g(num_samples) for g in src_gens]
            self.num_samples = num_samples

         @staticmethod
         def pred_error(x, y):
            return np.linalg.norm(y-x)**2 / (np.linalg.norm(x) * np.linalg.norm(y))

         def on_epoch_end(self, epoch, logs={}):
            m1 = self.modes[0].infer(self.num_samples)
            m2 = self.modes[1].infer(self.num_samples)
            self.mutual_information.append(mutual_info(m1, m2, 20))
            err = self.pred_error
            s = self.src
            self.pred_errors.append([
               err(m1, s[0]), err(m1, s[1]), err(m2, s[1]), err(m2, s[0]) ])

      self.sep_recorder = SeparationRecorder(self.modes, signal_gens)
      self.loss_recorder = LossRecorder()

      self.signal_gens = signal_gens


   def train(self, n_epochs, batch_size=128):
      train_batch(
         self.model, self.frames, self.frames,
         self.on_batch,
         batch_size, n_epochs,
         callbacks=[tools.Logger(), self.loss_recorder, self.sep_recorder]
      )

   def train_just_critic(self, n_epochs, batch_size=128):
      train_batch(
         self.model, self.frames, self.frames,
         lambda bs, _: self.critic.train(self, bs),
         batch_size, n_epochs,
         callbacks=[tools.Logger()]
      )



   def summary(self):
      code = self.encoder.predict(self.frames)
      pred_dep = self.critic.model.predict(code).mean()
      idx = np.arange(code.shape[0])
      np.random.shuffle(idx)
      code[:,:self.latent_sizes[0]] = code[idx,:self.latent_sizes[0]]
      pred_ind = self.critic.model.predict(code).mean()
      print(
         "code mean: {:.4f}, std: {:.4f}, dependent: {:.4f}, independent: {:.4f}".format(
            code.mean(), code.std(), pred_dep, pred_ind
         ))


def plot_modes3(sep, n=2000):
   figure()
   plot(sig1(n), 'k')
   plot(sig2(n), 'k')
   plot(build_prediction(sep.modes[0], sep.frames, n), 'r')
   plot(build_prediction(sep.modes[1], sep.frames, n), 'r')


def train_and_summary(sep, n_epochs, batch_size=128):
   sep.train(n_epochs, batch_size)
   training_summary(sep)
