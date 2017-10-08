import os
os.environ['KERAS_BACKEND'] = 'theano'

import keras
import keras.backend as B
import keras.layers as L
import keras.models as M
import numpy as N

x = B.placeholder(shape=(2,))
m = B.variable(N.array([[ 2.0 ],[3.0]]))
b = B.variable(N.array([[-1.0 ]]))
y = m*x + m.T*m   # expected:  x + 2*m
y_ = y.sum()

m_grad = B.gradients(y_,m)
m_grad_f = B.function([x], [m_grad])

print(m_grad_f([N.array([1337,42])]))



x1 = L.Input(shape=(2,))
x2 = L.Input(shape=(2,))
m = L.Dense(2, weights=[N.array([[2., 3.],[-1.,2.]]).T, N.array([-1.,0.])])
m2 = L.Dense(2, weights=[N.array([[2., 3.],[-1.,2.]]).T, N.array([-1.,0.])])
#m2 = L.Dense(1, weights=[N.array([[2.,-1.]]).T, N.array([-2.])])
l1 = m(x1)
l2 = m2(m(x2))
y = L.add([l1,l2])
model = M.Model(inputs=[x1,x2], outputs=y)
model.predict([N.array([[1,2]]),N.array([[3,4]])])
model.compile(optimizer='sgd', loss='mean_absolute_error')
model.fit([N.array([[1,2]]),N.array([[3,4]])], N.array([[1.,2.]]),1,100,verbose=0)
