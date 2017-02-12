#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N  = 100
dw = 2 * pi / N

k    =  3. * dw
damp =  6. * dw
#k    =  4. * dw
#damp =  -2. * dw



def ptridiag( left , middle , right ):
   matrix = diag( left[:-1] , -1)  +  diag( middle )  +  diag( right[1:] , +1 )
   matrix[-1,0] = left[0]
   matrix[0,-1] = right[-1]
   return matrix


f  = arange( N ) * dw
a  = exp( 1j*f + damp )
b  = ones(N) 

I = eye(len(a))
A = diag(a)
B = b / N

K = ptridiag( ones(N) , -2*ones(N) , ones(N) )
#K = ptridiag( ones(N) , zeros(N) , ones(N) )
#phi = exp( 5*dw*1.j )
#K = ptridiag( phi*ones(N) , -2*ones(N) , conj(phi)*ones(N) )

M = dot( A , I + k*K )
#M = A


def H ( A , B , z ):    # state space transfer function
#   return  linalg.solve( z * (I + k*K)  +  M  , B )
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
N_W = 2000
W = linspace(-pi,pi,N_W)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )

#S = linalg.solve( I , M )
#S = linalg.solve( I + k*K , M )
print sort( abs(eig(M)[0]) )


figure()
plot( W , abs(Hz)[:,:] ,'k' )
plot( W , abs(Hz[:,N/2]) , 'b' , linewidth = 3 )

figure()
plot( abs( np.sum( (exp(-24j*f) * Hz)[:,N/2:] , axis = 1 ) ) )

figure()
plot( -diff(unwrap(angle(Hz[:,N/2]))) * N_W / (2*pi) , 'k' )
plot( abs(Hz[:,N/2]) , 'b' )

show();

