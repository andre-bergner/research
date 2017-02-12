#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N = 128
k =  0.26
damp = -0.25


f = linspace( -pi , pi , N )
a = 1j*f + damp
b = ones(N)

I = eye(len(a))
A = diag(a)
B = b *  (2*pi)/N/pi# * k * damp

K = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)

M = A + k*K


def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = 1.5 * linspace(-pi,pi,1000)
W = W
Hz = array( map( lambda w: H(A,B,i*w) , W ) )

#print eig(M)[0]
print sort( real(eig(M)[0]) )


figure()
plot ( W , abs(Hz)[:,::2] ,'k' )

figure()
plot( W , abs(Hz[:,N/2]) , 'k' )

figure()
plot( abs( np.sum( (exp(-4j*f) * Hz)[:,:] , axis = 1 ) ) )

show();

