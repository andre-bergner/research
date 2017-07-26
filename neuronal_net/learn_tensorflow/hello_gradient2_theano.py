import theano as th
from pylab import *
import timer

x = th.tensor.dscalar()
y = th.tensor.dscalar()
a = th.tensor.dscalar()
b = th.tensor.dscalar()
l = th.tensor.nnet.sigmoid(a*x + b)

err = 0.5 * (y-l)**2
derr = th.tensor.grad(err, [a,b])

ax_f = th.function([x,a,b], l)
derr_f = th.function([x,y,a,b], derr)

eta = .5
a = 1.0
b = 0.5

def one_step():
   global a, b

   da, db = derr_f(1.,0.1,a,b)
   a -= eta*da
   b -= eta*db

   da, db = derr_f(0.,0.9,a,b)
   a -= eta*da
   b -= eta*db

   return [a,b]

with timer.Timer() as t:
   result = [ one_step() for n in range(1000) ]

plot(array(result))
