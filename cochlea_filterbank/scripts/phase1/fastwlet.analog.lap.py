#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N = 140
k =  0.15
damp = -0.05

damp = 0.5
k =  1



f = linspace( -pi , pi , N )
a = 1j*f + damp
b = ones(N)

mod_amount = 0.97
fmod = f  - mod_amount * sin(f)
dmod = 1. - mod_amount * cos(f)
a = 1j*fmod + damp*dmod
#b = 1./dmod
b = 0.5 + damp*dmod

I = eye(len(a))
A = diag(a)
B = b

L = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1) - 2*np.diag(np.ones(N))
#K = L * dmod**0.5
K = L * dmod

M = A + k*K


def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = 1.5 * linspace(-pi,pi,1000)
W = W  - mod_amount*sin(W)
Hz = array( map( lambda w: H(A,B,i*w) , W ) )

#S = linalg.solve( I , M )
print sort( real(eig(M)[0]) )


figure()
plot ( W , abs(Hz)[:,::2] ,'k' )

figure()
plot( W , abs(Hz[:,N/2]) , 'k' )

show();

