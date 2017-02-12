#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

N = 64
damp = -0.1
k =  0.13


f = arange( N ) * 2*pi/N   
a = exp( 1j*f + damp )
e = exp( -1j*f )
b = (1. - a*e)

I = eye(len(a))
A = diag(a)
B = b

K = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
K[-1,0] = 1
K[0,-1] = 1


E = []
Kappas = linspace( 0 , 1 , 10000 )
for k in Kappas:
   M = dot( A , I + k*K )
   S = linalg.solve( I , M )
   e = mean( abs(eig(S)[0]) )
   E.append(e)
E = array(E)

figure()
plot( Kappas , E , 'k' )

show();

