#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

N    =  128
damp = -0.03
k    =  0.16

I    = eye(N)

x = linspace( 0 , 1 , N/2 + 1 )
fmod = 0.2 * exp( 3.3 * x )
fmod = cumsum(concatenate(( [fmod[0]] , diff(fmod) * tanh(3.5*sin(linspace(0.05,pi-0.2,N/2))) )))

Sfmod = cumsum( fmod[:-1] )
Sfmod = Sfmod * pi / Sfmod[-1]

Sfmod = concatenate([  [0] , Sfmod , 2*pi - Sfmod[::-1][1:] ])
fmod  = concatenate([ fmod[:-1] , fmod[::-1][:-1] ])


a = exp( damp*fmod + 1j*Sfmod )
e = exp( -1j*Sfmod )

#a = exp( damp + 1j*f )
#e = exp( -1j*f )
#b = ones(N)

A = diag(a)
B = 1. - a*e
#B = b



D = diff(diff(eye(N+1)).T)
D -= diag(diag(D))
D[-1,0] = D[-2,-1]
D[0,-1] = D[1,0]

K = diag(ones(N-1),1) + diag(ones(N-1),-1) \
#    - diag(ones(N-4),4) - diag(ones(N-4),-4) - diag(ones(4),-N+4) - diag(ones(4),+N-4)
K[-1,0] = 1
K[0,-1] = 1

K = diag(fmod[1:],1) + diag(fmod[:-1],-1)
K[-1,0] = fmod[0]
K[0,-1] = fmod[-1]



M  = dot(A,I + k*K)


def H ( A , B , z ):    # state space transfer function
#   return  linalg.solve( z*I  +  A  , B )
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = linspace(-pi,pi,2000)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )


S = linalg.solve( I , M )

print sort( abs(eig(S)[0]) )
#print sort(abs(eig( dot( inv(I+d1*D) , dot(A,I+d2*D) ))[0]))


#E = []
#D2 = linspace( -10 , 10 , 1000 )
#for d2 in D2:
#   E.append( sort( abs(eig( dot(A,I + d2*D) )[0]) )[0] )



figure()
plot( W , abs(Hz)[:,:] ,'k' )
#semilogx( W , abs(Hz)[:,:] ,'k' )
#semilogy( W , abs(Hz)[:,:] ,'k' )
#plot ( W,abs( dot(D,Hz.T).T )[:,:] ,'k' )

figure()
plot( abs(Hz[:,.1*N/2]) , 'k' )
plot( abs(Hz[:,.3*N/2]) , 'k' )
plot( abs(Hz[:,.7*N/2]) , 'k' )
plot( abs(Hz[:,.9*N/2]) , 'k' )

show();

