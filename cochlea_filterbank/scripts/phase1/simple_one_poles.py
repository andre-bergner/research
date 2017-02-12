#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

def transfer_d( b , a , z ):
   return  b / (z-a)

def transfer_a( b , a , s ):
   return  b / (s-a)

w = linspace( -pi , pi , 1000 )

damp = 0.1
r = exp( -damp )
i  = 1.j

figure()

H_sum = 0
for n in arange( -2 , 3 ):
   freq = 2*pi*n
   H = transfer_a( damp , damp + i*freq , i*w )
   H_sum += H * (-1)**n
   plot( w , abs(H) , 'k' )

plot( w , abs(H_sum) , 'r' )


plot( w , abs(transfer_d( 1-r , r*exp(i*freq) , exp(i*w) )) , 'b' )


#for freq in linspace( -pi , pi , 5 ):
#   plot( w , abs(transfer_a( damp , damp + i*freq , i*w )) , 'k' )
#   plot( w , abs(transfer_d( 1-r , r*exp(i*freq) , exp(i*w) )) , 'r' )

ylim([0,1])

show()