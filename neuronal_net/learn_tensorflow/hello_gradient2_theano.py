import theano as th
from pylab import *
import timer

x = th.tensor.dscalar()
y = th.tensor.dscalar()
a = th.tensor.dscalar()
b = th.tensor.dscalar()
l = th.tensor.nnet.sigmoid(a*x + b)

#err = 0.5 * (y-l)**2
err = -(y*th.tensor.log(l) + (1.-y)*th.tensor.log(1.-l))
derr = th.tensor.grad(err, [a,b])

ax_f = th.function([x,a,b], l)
derr_f = th.function([x,y,a,b], [*derr,err])

eta = .5
a = 1.0
b = 0.5

def one_step():
   global a, b

   da, db, loss = derr_f(1.,0.1,a,b)
   a -= eta*da
   b -= eta*db

   da, db, loss = derr_f(0.,0.9,a,b)
   a -= eta*da
   b -= eta*db

   return [a,b,loss]

with timer.Timer() as t:
   result = array([ one_step() for n in range(1000) ])

print('error for input "1.": {}'.format(0.1-ax_f(1.,a,b)))
print('error for input "0.": {}'.format(0.9-ax_f(0.,a,b)))

figure()
subplot(211)
title("coefficients")
plot(result[:,:2])
subplot(212)
title("loss")
plot(result[:,2])

