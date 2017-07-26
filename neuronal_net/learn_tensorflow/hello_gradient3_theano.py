import theano as th
from pylab import *
import timer

def farray(a):
   return np.array(a,dtype=np.float32)


x = th.tensor.dvector()
y = th.tensor.dvector()
a = th.tensor.dmatrix()
b = th.tensor.dvector()
l = th.tensor.nnet.sigmoid(a.dot(x) + b)

err = ((y-l)**2).sum()
derr = th.tensor.grad(err, [a,b])

ax_f = th.function([x,a,b], l)
derr_f = th.function([x,y,a,b], derr)

eta = .5
a = np.random.randn(1,2)
b = np.random.randn(1)

def one_step():
   global a, b

   da, db = derr_f([1.,0.],[0.1],a,b)
   a -= eta*da
   b -= eta*db

   da, db = derr_f([0.,1.],[0.9],a,b)
   a -= eta*da
   b -= eta*db

   return [a.copy(),b.copy()]

with timer.Timer() as t:
   result = [ one_step() for n in range(1000) ]

result = [(x[0,0],x[0,1],y[0]) for x,y in result]
plot(array(result))
