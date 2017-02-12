#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N = 128
k =  0.15
damp = -0.05




f = linspace( -pi , pi , N )
a = 1j*f + damp
b = ones(N)

fmod = f  - 0.96*sin(f)
dmod = 1. - 0.96*cos(f)
a = 1j*fmod + damp*dmod
b = real(a)

I = eye(len(a))
A = diag(a)
B = b

#K = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
K = np.diag(dmod[:-1],1) + np.diag(dmod[1:],-1)
#K = -10j * ( np.diag(diff(a),1) + np.diag(diff(a),-1) )

M = A + k*K


def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = 1.5 * linspace(-pi,pi,1000)
W = W  - 0.96*sin(W)
Hz = array( map( lambda w: H(A,B,i*w) , W ) )

#S = linalg.solve( I , M )
print sort( real(eig(M)[0]) )


figure()
plot ( W , abs(Hz)[:,::2] ,'k' )

figure()
plot( W , abs(Hz[:,N/2]) , 'k' )

show();

