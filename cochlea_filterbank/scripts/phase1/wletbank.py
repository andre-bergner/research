#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

N = 64

f = arange( N ) * pi/N   +  0.5*pi/N

def f_mod(x,pm=1):  return    x + pm*1.2 * sin(x + pm*0.25*sin(x))      #  0.97 * sin(f)
def d_mod(x,pm=1):  return  ( 1.0 + pm*1.2 * cos(x + pm*0.25*sin(x))*(1. + pm*0.25*cos(x)) ) * (pi)/N

#def f_mod(x,pm=1):  return   x
#def d_mod(x):    return   (2*pi)/N

def working_hand_crafted_version():
   damp  =   5
   k     =   4
   fmod  =   f_mod(f)
   dmod  =   d_mod(f)

   return   fmod , damp * dmod , k * dmod


def ptridiag( left , middle , right ):
   matrix = diag( left[:-1] , -1)  +  diag( middle )  +  diag( right[1:] , +1 )
   matrix[-1,0] = left[0]
   matrix[0,-1] = right[-1]
   return matrix


fmod , dmod , kmod = working_hand_crafted_version()

a = exp( dmod + 1j*fmod )
#e = exp( -1j*fmod )

A = diag(a)
#B = (1. - a*e)
B = ones(N)

K = ptridiag( ones(N) , -2*ones(N) , ones(N) ) * kmod
K = K.T

I = eye(len(a))

S  = dot( A , I + K )

S = concatenate([ zeros((1,len(S))) , S.T , zeros((1,len(S))) ]).T
S[0,0] = S[0,-2]
S[0,-2] = 0
S[-1,-1] = S[-1,1]
S[-1,1] = 0

X = []
x = ones( len(S) )
for n in range(1024):
   x = concatenate([ [conjugate(x[0])] , x , [conjugate(x[-1])] ])
   x = dot( S , x )
   X.append( x )

X = array(X)
figure()
for n in range(N): plot( abs(fft(X[:,n])) , 'k'  )

figure()
plot( abs(fft(X[:,N/2])) , 'k'  )

show();

