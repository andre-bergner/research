import theano as th
from pylab import *
import timer

x = th.tensor.dscalar()
y = th.tensor.dscalar()
a = th.tensor.dscalar()

err = 0.5 * (y-a*x)**2
derr = th.tensor.grad(err, a)
derr_f = th.function([x,y,a], [derr,a*x])

eta = 0.01
a = 1.

def one_step():
   global a
   da, ax = derr_f(1.,0.,a)
   a -= eta * da
   return ax

with timer.Timer() as t:
   result = [ one_step() for n in range(1000) ]

plot(result)
