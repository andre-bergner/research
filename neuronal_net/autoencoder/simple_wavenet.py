# TODO 
# * predict outout probability distribution instead of single value
#   → use more features per layer
#   → cut before final layer to have more inputs and add dense layer
#   → use resiudal connections from previous layers (concat)
#   → visualize probability distribution over time

import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as B
from keras import regularizers

receptive_power = 9
receptive_depth = 2**receptive_power

def down_sample_conv():
   return L.Conv1D(8, kernel_size=(2), strides=(2), activation='relu', kernel_regularizer=regularizers.l2(0.001))

wavenet = M.Sequential()
wavenet.add(L.Reshape((receptive_depth,1), input_shape=(receptive_depth,)))
for _ in range(receptive_power):
   wavenet.add(down_sample_conv())
wavenet.add(L.Flatten())
wavenet.add(L.Dense(units=1, activation='tanh'))

wavenet.compile(loss='mean_squared_error', optimizer='adam')

# wavenet_f = B.function([wavenet.input],[l.output for l in wavenet.layers])
# print(wavenet_f([ np.matrix(np.arange(receptive_depth)) ])[-1])


t = np.arange(10000)
train_signal = np.sin(0.02*t + 4*np.sin(0.11*t) + 2*np.sin(0.009*t))

training_inputs = np.array([train_signal[n:n+receptive_depth] for n in range(len(train_signal)-receptive_depth) ])

wavenet.fit(training_inputs, train_signal[receptive_depth:], epochs=3, batch_size=20)
#, verbose=0, callbacks=[tools.Logger()] )

def predict_signal(length=1000):
   state = training_inputs[0]
   prediction = []
   for n in range(length):
      y = wavenet.predict(state.reshape(1,receptive_depth))[0,0]
      prediction.append(y)
      state[:-1] = state[1:]
      state[-1] = y
   return prediction


from pylab import *

plot(predict_signal())


# import sounddevice as sd
# sd.play(sound,44100)

