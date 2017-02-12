#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N = 141
k =  0.15
damp = -0.05

damp = 0.5
k =  1

f = linspace( -pi , pi , N )
a = 1j*f + damp
b = ones(N)

I = eye(len(a))
A = diag(a)
B = b / ( N/(2*pi) )

L = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1) - 2*np.diag(np.ones(N))
#K = L * dmod**0.5
K = L

M = A + k*K


def H ( A , B , s ):    # state space transfer function
   return  linalg.solve( s * I  +  M  , B )


i = 1.j
W = 1.5 * linspace(-pi,pi,1000)
Hz = array( map( lambda w: H(A,B,i*w) , W ) )

print sort( real(eig(M)[0]) )


u = ones(N) + 0.j
U = [u]
for n in range(4096):
   u = u + 0.01*dot( M , u )
   U.append(u)
U = array(U)



figure()
plot ( W , abs(Hz)[:,::2] ,'k' )

figure()
plot( W , abs(Hz[:,N/2]) , 'k' )

figure()
plot( W , -sqrt( max(log(abs(Hz[:,N/2]))) - log(abs(Hz[:,N/2]))) , 'k' )

figure()
plot( abs(U[:,N/2]) , 'k' )

show();

