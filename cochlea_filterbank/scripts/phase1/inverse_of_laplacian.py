#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
import scipy.linalg as la
from scipy.signal import *


N = 64
a = 0.2

I  = eye(N)


#L = diff(diff(eye(N+1)).T) + a*eye(N)  # Laplacian

L = diff(eye(N+1))[:N,:N] - .2*eye(N)
L = dot( L , L.T )
L[-1,-1] = L[0,0]    # correct (open!) boundaries

L_ = inv(L)

J = la.toeplitz( exp(-0.9*abs(linspace(-1,1,N))) )

plot( L_ , 'k' )




figure()
L  = diff( eye(N+1) )[:N]
L[0,-1] = 1
La = I - 8*L
La_ = inv( La )
plot( La_[:,::4] , 'k' )

a = -La[1,0] / La[0,0]
b = 1-a
c = 1 / ( 1 - a**N )   # correction for infentily many interactions


an_La_ = b * a**arange(N)    # analytic construction
# or alternatively...
x = zeros(N); x[0] = 1
an_La_ = lfilter( [c*b] , [1,-a] , x )

figure()
plot(La_[:,0] - an_La_)




figure()
L = diff(diff(eye(N+1)).T)
L[0,-1] = -1
L[-1,0] = -1
La = I - 8*L
La_ = inv( La )
plot( La_[:,::4] , 'k' )

a = -La[1,0] / La[0,0]
b = 1-a
c = 1 / ( 1 - a**N )   # correction for infentily many interactions


an_La_ = b * a**arange(N)    # analytic construction
# or alternatively...
x = zeros(N); x[0] = 1
an_La_ = lfilter( [c*b] , [1,-a] , x )

figure()
plot(La_[:,0] - an_La_)





N = 1024
A =  -(0.9 + 0.18*sin( arange( N ) * 2*pi/N ))
M =  eye(N) + diag(A[:-1],1) + diag([A[-1]],-N+1)
M_ = inv(M)

P = 1 / ( 1 - prod(A) )


#plot( P*roll(cumprod(concatenate(([1],roll(-A,-1)[:-1]))),1)  - M_[1,:]  )

E = []
for n in range(len(A)):
   E.append( P*roll(cumprod(concatenate(([1],roll(-A,-n)[:-1]))),n)  - M_[n,:] )
E = array(E)

figure()
imshow(E)






show()
