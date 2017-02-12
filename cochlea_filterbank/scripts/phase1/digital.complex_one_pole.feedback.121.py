#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N = 15
f = logspace( log10(0.1) , log10(1.9) , N )


damp = .4
a = exp( (1j-damp)*f )
b = (1 - exp((-damp)*f) )

I = eye(len(a))
A = diag(a)
B = b    # = diag(B) * ones(len(b))

D = concatenate(( b , [0] )); D = diff(diff( diag(D) ).T)

#D = diff(diff(eye(N+1)).T)# - 2*eye(N) + diag(B)
#D = diff(diff(diff(diff(eye(N+2)).T)).T)


def H ( A , B , z ):    # state space transfer function
   return  dot( inv( z*(0.6*I+(0.4)*D) - A - (0.3+0.j)*D ) , B )
#   return  dot( inv( z*z*I - z*A + 0.4*D ) , z*z*B )

i = 1.j
W = linspace(-pi,pi,2000)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )


"""

from scipy.signal import *

b , a  =  ellip ( 8 , 0.03 , 40 , 0.5 )
_,down_2 =  freqz ( b , a , W )

for n in range(N):
   Hz[:,n] *= down_2

"""

plot ( W,abs(Hz)[:,:] ,'k' )
#semilogy ( abs(Hz)[:,1:-1] ,'k' )


"""
for n in range(N):
   Hz[:,n] *= exp( -i*.09*n )       # N=150, damp=0.6
   Hz[:,n] *= 1+0.25*f[n]
#   Hz[:,n] *= exp( -i*.24*n )       # N=50, damp=0.6

#Hz[:,60:80:] *= 0

Hsum  =  np.sum(Hz[:,2:-2],axis=1)

plot ( W, 0.053*abs(Hsum)  ,'r' )
#semilogy ( W, abs(Hsum)  ,'r' )
"""

show();

