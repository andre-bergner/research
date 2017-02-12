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

N  =  120

I    = eye(N)
f = arange( N ) * 2*pi/N   +  pi/N



def local_periodic_version():
   global N
   damp = -0.8
   k    =  30
   fmod = pi + 1.01 + sin(f) + sin(3*f)/9
   fmod_ = concatenate(( [fmod[-1]] , fmod , [fmod[0]] ))
   dmod = 0.001 + abs( 0.5 * ( fmod_[2:] - fmod_[0:-2] ) )
   foo  = (0.001 + dmod)**2
   return   fmod , damp*dmod , k*foo



def f_mod(x,pm=1):  return   x + pm*1.1 * sin(x + pm*0.25*sin(x))      #  0.97 * sin(f)
def d_mod(x):    return   ( 1.0 + 1.1 * cos(x + 0.25*sin(x))*(1. + 0.25*cos(x)) ) * (2*pi)/N


def working_hand_crafted_version():
   damp  =  -0.5
   k     =   2.5
   fmod  =   f_mod(f)
   dmod  =   d_mod(f)

   return   fmod , damp * dmod , k * dmod


   #fmod_  =  concatenate(( [fmod[-1]-2*pi] , fmod , [fmod[0]+2*pi] ))
   #dmod   =  0.5 * ( fmod_[2:] - fmod_[0:-2] )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def ptridiag( left , middle , right ):
   matrix = diag( left[:-1] , -1)  +  diag( middle )  +  diag( right[1:] , +1 )
   matrix[-1,0] = left[0]
   matrix[0,-1] = right[-1]
   return matrix

def rolldiag( M , diag_nr , roll_amount ):
   d = concatenate(( diag( M , diag_nr ) , diag( M , -(len(M)-diag_nr) ) ))
   d = roll( d , roll_amount )
   Mtemp = diag( d[:-diag_nr] , diag_nr )  +  diag( d[len(M)-diag_nr:] , -(len(M)-diag_nr) )
   return  M * (Mtemp == 0)  +  Mtemp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


fmod , dmod , foo = working_hand_crafted_version()

a = exp( dmod + 1j*fmod )
e = exp( -1j*fmod )

A = diag(a)
#B = (1. - a*e) / abs(dmod)**2.5
B = ones(N)# * foo

K = ptridiag( foo , zeros(N) , foo )



M  = dot( A , I + K )
#M  = rolldiag( M , 1 , -5 )

def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = f_mod(linspace(-pi,pi,1000),-1)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )

#B /= np.max( abs(Hz) , axis = 0 )
#Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )


S = linalg.solve( I , M )
E = sort( abs(eig(S)[0]) )
print "eigenvalues\n", E



figure()
subplot(211)
plot( W , abs(Hz)[:,:] ,'k' )
subplot(212)
plot( W*22050/pi , abs(Hz)[:,N/2::4] ,'k' )

figure()
subplot(211)
plot( W , abs(Hz)[:,:] / np.max(abs(Hz),axis=0) ,'k' )
subplot(212)
plot( abs(Hz)[:,:] / np.max(abs(Hz),axis=0) ,'k' )

figure()
imshow( (abs(Hz)[:,:] / np.max(abs(Hz),axis=0))[::-1] , aspect='auto')



figure()
W = f_mod(linspace(-pi,pi,N),-1)
Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )
Hn = abs(Hz)[:,:] / np.max(abs(Hz),axis=0)

plot( diff(diag(abs(Hn))) )
plot( diff(diag(abs(Hn),1)) )  # <-- minimize this !!!
plot( diff(diag(abs(Hn),2)) )  # <-- and this !!!


show()
