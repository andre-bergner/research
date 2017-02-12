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

N = 128

circular = True


d1 =  0.5
d2 = -0.2




if circular:

   f = arange( N ) * 2*pi/N
   fmod = 0.1*sin(f)
   damp = 1
   
   damp = +0.8
   d1 =  -0.1
   d2 = -.28

   damp = +.2
   d1 = 0.
   d2 = -.1


   #damp = .2
   #d1 = 0
   #f = linspace( 0  , pi , N )

else:

   f = arange( N ) * 1*pi/N
   damp = .4


a = exp( 1j*f + damp )
b = ones(N)

I = eye(len(a))
A = diag(a)
B = b

D = diff(diff(eye(N+1)).T)

if circular:
   D[-1,0] = D[-2,-1]
   D[0,-1] = D[1,0]

else: 
   D[0,0] = -D[0,1]
   D[-1,-1] = -D[-2,-1]

   win = 1 - exp( -4*sin(linspace( 0 , pi , N )) )
   Win = diag(win)
   #D = dot( Win , D )
   A = dot( Win , A )


Bk  = I + d1*D          #
Fw  = dot(A,I + d2*D)   # forward matrix


def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * Bk  +  Fw  , B )
#   return  linalg.solve( z * (I+d1*D)  +  dot(A,I+d2*D)  , B )


i = 1.j
W = linspace(-pi,pi,2000)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )


S = linalg.solve( Bk , Fw )

print sort( abs(eig(S)[0]) )
#print sort(abs(eig( dot( inv(I+d1*D) , dot(A,I+d2*D) ))[0]))


#E = []
#D2 = linspace( -10 , 10 , 1000 )
#for d2 in D2:
#   E.append( sort( abs(eig( dot(A,I + d2*D) )[0]) )[0] )


"""
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
"""

figure()
plot ( W,abs(Hz)[:,:] ,'k' )
#plot ( W,abs( dot(D,Hz.T).T )[:,:] ,'k' )
#semilogy ( abs(Hz)[:,1:-1] ,'k' )

figure()
plot( abs(Hz[:,N/2]) , 'k' )

show();

