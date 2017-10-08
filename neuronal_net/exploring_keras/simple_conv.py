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

model1 = M.Sequential()
model1.add(L.Reshape((4,1), input_shape=(4,)))
model1.add(L.Conv1D(1, kernel_size=(2), strides=(2), weights=[np.ones((2,1,1)), np.zeros(1)]))
model1.add(L.Conv1D(1, kernel_size=(2), strides=(2), weights=[np.ones((2,1,1)), np.zeros(1)]))
model1.add(L.Flatten())

model1_f = B.function([model1.input],[model1.layers[1].output,model1.layers[2].output,model1.output])
print("Using Strides")
print(model1_f([ np.matrix([1,2,3,4]) ]))


model2 = M.Sequential()
model2.add(L.Reshape((4,1), input_shape=(4,)))
model2.add(L.Conv1D(1, kernel_size=(2), dilation_rate=(1), weights=[np.ones((2,1,1)), np.zeros(1)]))
model2.add(L.Conv1D(1, kernel_size=(2), dilation_rate=(2), weights=[np.ones((2,1,1)), np.zeros(1)]))
model2.add(L.Flatten())

model2_f = B.function([model2.input],[model2.layers[1].output,model2.layers[2].output,model2.output])
print("Using Dilations")
print(model2_f([ np.matrix([1,2,3,4]) ]))


