import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras
import keras.models as M
import keras.layers as L
import keras.backend as B

"""
model = M.Sequential()
model.add(L.Dense(units=2, input_dim=4))
model.add(L.Activation('sigmoid'))

model_f = B.function([model.input],[model.output])
model_f([ np.ones((1,4)) ])
"""


model = M.Sequential()
model.add(L.Reshape((4,1), input_shape=(4,)))
model.add(L.Conv1D(1, kernel_size=(2), strides=(2), weights=[np.ones((2,1,1)), np.zeros(1)]))
model.add(L.Conv1D(1, kernel_size=(2), strides=(2), weights=[np.ones((2,1,1)), np.zeros(1)]))
model.add(L.Flatten())
#model.add(L.Activation('sigmoid'))

model_f = B.function([model.input],[model.layers[1].output,model.layers[2].output,model.output])
print(model_f([ np.matrix([1,2,3,4]) ]))


