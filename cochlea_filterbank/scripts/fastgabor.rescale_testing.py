#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

i = 1.j


#
#  TTTTT  OOO   DDDD    OOO       - try Laplacian vs Adjacency matrix
#    T   O   O  D   D  O   O      - harvest eigenvalues
#    T   O   O  D   D  O   O      - test with b-normalization
#    T    OOO   DDDD    OOO       - more plots
#
#

def transfer_function ( A , B , z ):    # state space transfer function
   I = eye(len(A))
   return  linalg.solve( z * I  +  A  , B )


def  compute_gaborbank( N , damp , k , range , N_range ):

   f = arange( N ) * 2*pi/N   
   a = exp( 1j*f + damp )
   e = exp( -1j*f )
   b = ones(N) #/ N
#   b =  (1. - a*e) #/ N

   A0 = diag(a)
   B  = b
   K  =  -2*np.diag(np.ones(N)) + np.diag(np.ones(N-1),1)  +  np.diag(np.ones(N-1),-1)
   K[-1,0] = 1
   K[0,-1] = 1
   A = dot( A0 , eye(N) + k*K )

   W = linspace( -range , range , N_range )
   Hz = array( map( lambda w: transfer_function( A , B , exp(i*w) )  ,  W ) )
   #EigenValue =  mean( abs( eig( linalg.solve(eye(N),A) )[0]) )

   return  Hz[:,N/2] #, EigenValue  # return just the center one, as all are symmetric



damp = -5.12 * -8
range = 12.8 * 10    # 128
k =  25.6

N_range = arange( 24 , 513 , 4 )
N_range = arange( 24 , 130 , 4 )
dots = 256

info_vs_N = []
H_collection = []
AbsEigenValues = []
figure()
for N in N_range:
   H = compute_gaborbank( N , damp/N , k/N , min( range/N , pi ) , dots )
   H_collection.append( H )
   #AbsEigenValues.append( E )
   print N
   plot( abs(H) , 'k' )
   H_max = abs( H )
   H_sum = sum(abs( compute_gaborbank( N , damp/N , k/N , min( range/N , pi ) , dots ) ))
   info_vs_N.append(( H_max[0] , H_sum / dots ))
info_vs_N = array(info_vs_N)


savetxt( "gaborbank.info.vs.N.txt" , array([ N_range , 2*pi / N_range , info_vs_N[:,1] , info_vs_N[:,0] ]) )


figure ()
plot   ( 2*pi / N_range , info_vs_N[:,0] , 'k' )
xlabel ( r'$\Delta\omega$' , fontsize = 20 )
ylabel ( r'max' , fontsize = 20 )
title  ( 'maximum of impulse response')


figure ()
plot   ( 2*pi / N_range , info_vs_N[:,1] / info_vs_N[:,0] , 'k' )
xlabel ( r'$\Delta\omega$' , fontsize = 20 )
ylabel ( r'$\int d\omega$' , fontsize = 20 , rotation = 0)
title  ('normalized integral')

figure()
loglog(  2*pi / N_range[3:] , info_vs_N[3:,0] , 'k' )
loglog(  2*pi / N_range[3:] , info_vs_N[3:,1] , 'r' )

figure()
plot( diff(log( diff(info_vs_N[3:,0]) / (info_vs_N[3:-1,0]) )) / diff(log(N_range[2:-2])) , '.' )

figure()
plot( log(N_range[3:-1]) , log( diff(info_vs_N[3:,0]) / info_vs_N[3:-1,0] )  , '.' )

# N = 256
# H_coup = ( compute_gaborbank( N , damp/N , k/N , pi , 1024 ) )
# H_pure = ( compute_gaborbank( N , damp/N ,  0  , pi , 1024 ) )
# plot( H_coup / H_coup[150] , 'k' )


show()
