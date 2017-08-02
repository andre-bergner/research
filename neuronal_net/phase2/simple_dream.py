import os
os.environ["KERAS_BACKEND"] = "theano"

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K


proto_input = [ [1,0,
                 1,0]
              , [1,1,
                 0,0]
              , [1,0,
                 0,1]
              , [0,1,
                 1,0]
              ]

def normalize2(M):
   scale = 1. / (np.max(M)-np.min(M))
   ofs = np.min(M)
   return (M - ofs) * scale

def normalize(M):
   return M / np.max(M)

def dirac(n,s):
   d = np.zeros(s)
   d[n] = 1.
   return d

inputs = np.matrix([ normalize(proto_input[n%4] + np.abs(0.1*np.random.randn(4))) for n in range(4*500) ])
features = np.matrix([ dirac(n%4,4) for n in range(4*500) ])


model = Sequential()
model.add(Dense(units=4,input_dim=4))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1.0))
model.fit(inputs, features, epochs=100, batch_size=20)

correct_results = [ np.argmax(num) == np.argmax(model.predict(img)) for img, num in zip(inputs,features) ]
print("correct results: {0:.2f} %".format( 100. * float(np.count_nonzero(correct_results)) / len(correct_results) ) )


#   -------------------------------------------------------------------------------------
#  THE DREAM
#   -------------------------------------------------------------------------------------

from keras import backend as K

feat = K.placeholder(shape=(10,))
loss = ((model.output-feat)**2).sum()
dream_grad = K.gradients(loss, model.input)
dream_grad_f = K.function([model.input, feat], [dream_grad])

dream = np.random.randn(1,4)
dream_orig = dream.copy()
#ddream = dream_grad_f([dream, [1,0,0,0]])
for _ in range(100):
   ddream = dream_grad_f([dream, [1,0,0,0]])[0]
   dream = normalize2(dream - 0.1*ddream)









