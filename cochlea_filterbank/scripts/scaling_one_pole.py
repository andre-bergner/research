#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
import scipy.linalg as la
from scipy.signal import *

np.set_printoptions(linewidth=140)


i = 1.j
N = 4000
w = linspace(-pi,pi,N)
z = exp(i*w)

def H(z,a,b):  return b / ( a - z )

phi = 1
a = 0.8 * exp( i * phi )
b = 0.2

scale = 2
phi2 = 0.2


def pole_transform( z , old_phase , new_phase , scale ):
   return  scale * ( exp(i*new_phase)*z - exp(i*old_phase) ) + exp(i*old_phase) 

#  TODO ---------------------------------------------
# 
#  express transform of transfer function in terms of
#  mappings of the parameters a and b.


plot( w , abs(H(z,a,b)) , 'k' , linewidth=2 )

for new_phi in arange( 0.1, 2 , 0.3 ):
   plot( w , abs(H( pole_transform(z, phi, new_phi, scale) , a , b )) , 'b' , alpha=0.6 )

for scale in arange( 0.5, 3 , 0.4 ):
   plot( w , abs(H( pole_transform(z, phi, new_phi, scale) , a , b )) , 'g' , alpha=0.6 )

xticks( [-pi,0,pi] , [r'$-\pi$' , 0 , r'$\pi$'])
xlim(( -pi , pi  ))
ylim((   0 , 1.2 ))


show()
