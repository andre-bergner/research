import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import keras.layers as L
import keras.backend as K

len = 3
i = L.Input((4,1,))
p = L.ZeroPadding1D((0,len-1))
c = L.Conv1D(1, padding='valid', kernel_size=len, use_bias=False, weights=[np.array([[len*[1]]]).T], strides=2)
f = K.function([i],[p(i),c(p(i))])
#c = L.Conv1D(1, padding='same', kernel_size=3, use_bias=False, weights=[np.array([[[1,1,0]]]).T], strides=2)
#f = K.function([i],[c(i)])
f([np.reshape([1,2,3,4],(1,4,1))])
