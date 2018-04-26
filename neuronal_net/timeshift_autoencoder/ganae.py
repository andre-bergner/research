import keras
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

class GAN():
    def __init__(self):

        self.z_size = 8
        self.img_shape = (28, 28, 1)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mean_squared_error', optimizer=optimizer)#, exception_verbosity='high')


        # Build and compile combined
        x = Input(shape=self.img_shape)
        img = self.generator(x)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # an image as input => auto-encodes images => determines validity 
        self.combined = Model(x, valid)
        #self.combined.layers[1].layers[-2].activity_regularizer = \
        #    lambda a: K.sum(keras.losses.mean_squared_error(self.combined.layers[1].layers[1],a))
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)#, exception_verbosity='high')


    def build_generator(self):

        x = Input(shape=self.img_shape)

        flatten = Flatten()(x)

        dense1 = Dense(480)(flatten)
        act1 = LeakyReLU(alpha=0.2)(dense1)
        batch1 = BatchNormalization(momentum=0.8)(act1)

        dense2 = Dense(160)(batch1)
        act2 = LeakyReLU(alpha=0.2)(dense2)
        batch2 = BatchNormalization(momentum=0.8)(act2)

        dense3 = Dense(self.z_size)(batch2)
        act3 = LeakyReLU(alpha=0.2)(dense3)
        batch3 = BatchNormalization(momentum=0.8)(act3)

        dense4 = Dense(160)(batch3)
        act4 = LeakyReLU(alpha=0.2)(dense4)
        batch4 = BatchNormalization(momentum=0.8)(act4)

        dense5 = Dense(480)(batch4)
        act5 = LeakyReLU(alpha=0.2)(dense5)
        batch5 = BatchNormalization(momentum=0.8)(act5)

        final_dense = Dense(np.prod(self.img_shape), activation='tanh')
        #final_dense.activity_regularizer = lambda a: K.sum(keras.losses.mean_squared_error(flatten,a))
        dense6 = final_dense(batch5)
        reshape = Reshape(self.img_shape)(dense6)

        model = Model(x, reshape)
        model.summary()

        return model


    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(480))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(160))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        self.X_train = X_train

        half_batch = batch_size // 2


        def train_discriminator():

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Auto-encode a half batch of images
            ae_imgs = self.generator.predict(imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(ae_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            return d_loss


        def train_generator():

            ### select new images ???
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            self.generator.train_on_batch(imgs, imgs)
            g_loss = self.combined.train_on_batch(imgs, valid_y)

            return g_loss


        #print("Pre-train Auto-encoder ------------------------------")
        #self.generator.fit(X_train, X_train, batch_size=batch_size, epochs=2)

        #print("Pre-train Discriminator -----------------------------")
        #for n in range(5000):
        #    d_loss = train_discriminator()
        #    print("%d [D loss: %f, acc.: %.2f%%]" % (n, d_loss[0], self.z_size*d_loss[1]))


        for epoch in range(epochs):

            d_loss = train_discriminator()
            g_loss = train_generator()

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], self.z_size*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)


    def save_imgs(self, epoch):
        r, c = 5, 3
        idx = np.random.randint(0, self.X_train.shape[0], r * c)
        imgs = self.X_train[idx]

        gen_imgs = self.generator.predict(imgs)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, 2*c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,2*j].imshow(imgs[cnt, :,:,0], cmap='gray')
                axs[i,2*j].axis('off')
                axs[i,2*j+1].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


gan = GAN()
gan.train(epochs=30000, batch_size=32, save_interval=200)

