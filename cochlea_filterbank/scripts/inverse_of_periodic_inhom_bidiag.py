#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
from scipy.signal import *

np.set_printoptions(linewidth=140)

def scanl( f, y, xs , zs ):
    for x,z in zip(xs,zs):
        y = f(y,x,z)
        yield y

def circulant_one_pole( xs , zs ):
    y     = array( list( scanl( lambda y,x,a: a*y - x , 0     , xs   , zs )))
    y_inf = array( list( scanl( lambda y,x,a: a*y - x , y[-1] , xs*0 , zs ))) # compute the tail
    return y + y_inf * 1 / (1 - prod(zs))                      # add tail with factor compensating the infinite repetition

N = 512

x = zeros(N)
x[0] = 1

#   ----------------------------------------------------------------------------
#  Circulant discrete Nabla
#   ----------------------------------------------------------------------------

f  = arange(N) * 2 * pi / N
a  = exp(0.1j*f - 0.01*f)
M  = diag(a) - diag(ones(N-1), 1);
M[-1,0] = -1
M_ = inv(M)

y = roll(circulant_one_pole(x, a), 1)

figure()
subplot(211)
plot(M_[:,0], 'k')
plot(y, 'r')

subplot(212)
plot(M_[:,0] - y, 'k')

#   ----------------------------------------------------------------------------
#  Circulant discrete Laplace
#   ----------------------------------------------------------------------------

L = 1.4*diag(.5-0.05*a) - diag(ones(N-1), 1) - diag(ones(N-1), -1)
L[-1,0] = -1
L[0,-1] = -1
L_ = inv(L)

figure()
subplot(211)
plot(L_[:,0], 'k')
# plot(y , 'r')



show()
