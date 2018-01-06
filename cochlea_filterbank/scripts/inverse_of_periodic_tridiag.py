#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
import scipy.linalg as la
from scipy.signal import *

np.set_printoptions(linewidth=140)


def circulant_one_pole( a , x ):
   y,z     = lfilter( [1] , [1,a] , x , zi = [0] )
   y_inf,z = lfilter( [1] , [1,a] , zeros(len(x)) , zi = z )  # compute the tail
   return y + y_inf * 1 / ( 1 - a**len(x) )                   # add tail with factor compensating the infinite repetition


N = 512

x = zeros(N)
x[N//4] = 1


#   ---------------------------------------------------------------------------------------------
#  Circulant discrete nabla
#   ---------------------------------------------------------------------------------------------

k = -0.999

v = zeros(N);  v[:2] = [1,k]
Nab = la.circulant(v)
Nab_ = inv(Nab)

y = circulant_one_pole( k , x )

figure()
subplot(211)
plot( log(Nab_[:,N//4]) , 'k' )
plot( log(y) , 'r' )

subplot(212)
plot( log(Nab_[:,N//4]) - log(y) , 'k' )






#   ---------------------------------------------------------------------------------------------
#  Circulant discrete Laplacian
#   ---------------------------------------------------------------------------------------------

#k = -0.499
k = -0.10472

a = 0.5/k + sqrt( 0.25/(k*k) - 1 )
g = 1 / (2 / (1+a*a) - 1)

v = zeros(N);  v[:2] = [1,k];  v[-1] = k
Lap = la.circulant(v)
Lap_ = inv(Lap)


y1 = circulant_one_pole( a , x )
y2 = circulant_one_pole( a , x[::-1] )[::-1]
y = g*(y1 + y2 - x)


figure()

subplot(211)
plot( log((Lap_[:,N//4])) , 'k' )
plot( log((y)) , 'r' )

subplot(212)
plot( log((Lap_[:,N//4])) - (log(y)) , 'k' )

title( "1. circulant one pole" )



half = lambda x: x[:len(x)//2]

def circulant_one_pole_mirrored( a , x ):
   N = len(x)
   #x = half(x)

   y,z     = lfilter( [1] , [1,a] , x , zi = [0] )
   y_inf,z = lfilter( [1] , [1,a] , zeros(len(x)) , zi = z )

   factor_delay = a**N
   factor_inf   = 1 / ( 1 - a**(2*N) )

   return   y  +  conj(y_inf[::-1]) * factor_inf  +  y_inf * factor_delay * factor_inf  

y1 = circulant_one_pole_mirrored( a , half(x)*1.j )
y2 = circulant_one_pole_mirrored( a , half(x)[::-1]*1.j )[::-1]
y = g*(imag(y1) + imag(y2) - half(x))

figure()
subplot(211)
plot( half(dot(Lap_, x-x[::-1] )) - y , 'k' )
title( "2. mirrored circulant one pole " )

subplot(212)
plot( log(abs( half(dot(Lap_, x-x[::-1] )) )) - log(abs(y)) , 'k' )
title( "2. mirrored circulant one pole (log-scale)" )


"""
#   ---------------------------------------------------------------------------------------------
#  Open discrete laplacian
#   ---------------------------------------------------------------------------------------------

k = -0.49

v = zeros(N);  v[:2] = [1,k];  v[-1] = k
Lap = la.circulant(v)
Lap[-1,0] = Lap[0,-1] = 0
#Lap[0,1] = Lap[1,0] = Lap[-2,-1] = Lap[-1,-2] = -sqrt(-k)
Lap_ = inv(Lap)


x = zeros(N)
x[N/4] = 1

a = 0.5/k + sqrt( 0.25*k**-2 - 1 )
g = 1 / (2 / (1+a*a) - 1)

y1 = lfilter( [1] , [1,a] , x )
y2 = lfilter( [1] , [1,a] , x[::-1] )[::-1]
y = g*(y1 + y2 - x)


figure()

subplot(211)
plot( log(Lap_[:,N/4]) , 'k' )
plot( log(y) , 'r' )

subplot(212)
plot( log(Lap_[:,N/4]) - log(y) , 'k' )
"""




show()
