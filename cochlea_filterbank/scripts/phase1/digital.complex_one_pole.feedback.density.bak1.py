#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  TODO
#  - playing more with window functions
#  - analytically find eigenvalues of circular system
#  - analytically play with 3 pole system --> eigenvalues , stability
#  - how does an impuls inserted in one channel travel throught the net?
#    --> where do instabilities occur in the open system?

from pylab import *
np.set_printoptions(linewidth=140)

N    =  128     #  64
damp = -0.08     # -0.12
k    =  0.18     #  0.5
I    = eye(N)


f = arange( N ) * 2*pi/N
f2 = arange( N+1 ) * 2*pi/(N+1)

### loglog( 1.0001 + cos(f3 + 0.6*sin(f3)  ) , '.' )

#sine = sin(f)
#sine = sin( f + .6*sin(f))
#fmod = 1. + 0.995*tanh(2*sine)

#sine = cos(f2)
#fmod = 1.3 + 1.2995*tanh(2*sine)

sine = cos(f2 + 0.6*sin(f2) )
fmod = 1.0001 + sine



Sfmod = cumsum( fmod )
fmod = diff(Sfmod)
Sfmod = Sfmod - Sfmod[0]
Sfmod = Sfmod * 2*pi / Sfmod[-1]
Sfmod = 0.5 * (Sfmod[1:] + Sfmod[:-1])


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

pau = 1.1

K = diag(fmod[1:]**pau,1) + diag(fmod[:-1]**pau,-1)
K[-1,0] = fmod[0]**pau
K[0,-1] = fmod[-1]**pau



M  = dot(A,I + k*K)


def H ( A , B , z ):    # state space transfer function
#   return  linalg.solve( z*I  +  A  , B )
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = linspace(-pi,pi,5000)

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
#plot ( W,abs( dot(D,Hz.T).T )[:,:] ,'k' )

figure()
plot( abs(Hz[:,1*N/4]) , 'k' )
plot( abs(Hz[:,2*N/4]) , 'k' )
plot( abs(Hz[:,2.5*N/4]) , 'k' )
plot( abs(Hz[:,3*N/4]) , 'k' )

show();




N = 1024
#A = 1 + 0.96 * exp( 1.j * arange( N ) * 2*pi/N )
A =  0.9 + 0.18*sin( arange( N ) * 2*pi/N )**3
X = zeros(1024)
X[20] = 1
y = 0
Y = []
for xa in zip(X,A):
   y = xa[1]*y + xa[0]
   Y.append(y)

Y = array(Y)

