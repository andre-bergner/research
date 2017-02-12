#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

N = 512

damp = +0.5
d = -.2

damp = +0.35
d = -.45

damp = +0.08
d = -.15


f = linspace( 0  , pi , N )

a = exp( 1j*f + damp )
I = eye(len(a))
A = diag(a)

D = diff(diff(eye(N+1)).T)
D[-1,0] = D[-2,-1]
D[0,-1] = D[1,0]

S  = dot( A , I + d*D )

S = concatenate([ zeros((1,len(S))) , S.T , zeros((1,len(S))) ]).T
S[0,0] = S[0,-2]
S[0,-2] = 0
S[-1,-1] = S[-1,1]
S[-1,1] = 0

X = []
x = ones( len(S) )
for n in range(1024):
   x = concatenate([ [conjugate(x[1])] , x , [conjugate(x[-2])] ])
   x = dot( S , x )
   X.append( x )

X = array(X)
figure()
for n in range(N): plot( abs(fft(X[:,n])) , 'k'  )

figure()
plot( abs(fft(X[:,N/2])) , 'k'  )

show();

