#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

N  =  160

I    = eye(N)
f = arange( N ) * 2*pi/N   +  pi/N



def local_periodic_version():
   damp =  4    #-0.8
   k    =  5      # 30
   fmod = pi + 1.01 + 0.5*sin(f)# + sin(3*f)/9
   fmod_ = concatenate(( [fmod[-1]] , fmod , [fmod[0]] ))
   dmod = 0.1  +  0.5 * abs( fmod_[2:] - fmod_[0:-2] )
   return   fmod , damp*dmod , k*dmod
#   foo  = (0.01 + dmod)**2
#   dmod = 0.01 + abs( 0.5 * ( fmod_[2:] - fmod_[0:-2] ) )
#   return   fmod , damp*dmod , k*foo



def f_mod(x,pm=1):  return   x + pm*1.2 * sin(x + pm*0.25*sin(x))      #  0.97 * sin(f)
def d_mod(x):    return   ( 1.0 + 1.2 * cos(x + 0.25*sin(x))*(1. + 0.25*cos(x)) ) * (2*pi)/N
#   damp  =   6
#   k     =   4


#def f_mod(x,pm=1):  return   x + pm*0.97*sin(x)
#def d_mod(x):       return   (1.0 + 0.97*cos(x)) * (2*pi)/N
#   damp  =   6
#   k     =   5

def working_hand_crafted_version():
   damp  =   6#3 #5
   k     =   5#2 #4
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


#fmod , dmod , kmod = working_hand_crafted_version()
fmod , dmod , kmod = local_periodic_version()

a = exp( dmod + 1j*fmod )
e = exp( -1j*fmod )

A = diag(a)
B = (1. - a*e)
#B = ones(N)

#K = ptridiag( kmod , zeros(N) , kmod )
K = ptridiag( ones(N) , -2*ones(N) , ones(N) ) * kmod
K = K.T

M  = dot( A , I + K )
#M  = rolldiag( M , 1 , -5 )

def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * I  +  M  , B )

N_W = 1000
i = 1.j
W = f_mod(linspace(-pi,pi,N_W),-1)
#W = linspace(-pi,pi,N_W)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )

#B /= np.max( abs(Hz) , axis = 0 )
#Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )


print  "eigenvalues\n",  sort( abs(eig(M)[0]) )



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



#### RECONSTRUCTION ############

# what does work...
figure()
e = exp( -1j * arange(0,N) * 2*pi/N )
plot( abs(np.sum( (e**34 * Hz)[:,:] , axis = 1 )) , 'k' )
plot( abs(np.sum( (e**34 * Hz)[:,N/2:] , axis = 1 )) , 'r' )

# what does not work...
# e = exp( 1j*fmod )
# ...

# ---- assumption ---------
# need to compute  individual group delay of each bin/channel
# rotate each channel by it's group delay



#for 
#plot( -diff(unwrap(angle(Hz[:,3*N/4]))) * N_W / (2*pi) , 'k' )



"""
figure()
W = f_mod(linspace(-pi,pi,N),-1)
Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )
Hn = abs(Hz)[:,:] / np.max(abs(Hz),axis=0)

plot( diff(diag(abs(Hn))) )
plot( diff(diag(abs(Hn),1)) )  # <-- minimize this !!!
plot( diff(diag(abs(Hn),2)) )  # <-- and this !!!
"""

show()
