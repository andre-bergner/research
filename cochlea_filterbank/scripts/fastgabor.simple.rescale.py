#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N  = 100
dw = 2 * pi / N

k1   =  3. * dw
damp =  6. * dw
#k    =  4. * dw
#damp =  -2. * dw

k2 = -1.7*k1
k1 = 0


def ptridiag( left, middle, right ):
    matrix = diag( left[:-1], -1)  +  diag( middle )  +  diag( right[1:], +1 )
    matrix[-1,0] = left[0]
    matrix[0,-1] = right[-1]
    return matrix


f  = arange( N ) * dw
a  = exp( 1j*f + damp )
b  = ones(N) 

I = eye(len(a))
A = diag(a)
B = b / N

K = ptridiag( ones(N), -2*ones(N), ones(N) )
#K = ptridiag( ones(N), zeros(N), ones(N) )
#phi = exp( 5*dw*1.j )
#K = ptridiag( phi*ones(N), -2*ones(N), conj(phi)*ones(N) )

M1 = dot( A, I + k1*K )
M2 = I + k2*K

S = linalg.solve( M2, M1 )
print(sort(abs(eig(S)[0])))


def pole_transform( z, old_phase, new_phase, scale ):
    return scale * ( exp(i*new_phase)*z - exp(i*old_phase) ) + exp(i*old_phase) 


def H(A, B, z, s, phi=0):
    return linalg.solve(pole_transform(z, phi, 0, s) * M2 + M1, B)

#  IDEA -----------------------------------------
#
#  • vector-valued linear interpolation between scaled points
#  • each channel must be one at it's scales pos (max) and
#    zero at the scaled points of each of the other channels,
#    respectively


i = 1.j
N_W = 4000
W = linspace(-pi,pi,N_W)

Hz   = array( map( lambda w: H( A,B,exp(i*w), 1   ), W ) )
Hz12 = array( map( lambda w: H( A,B,exp(i*w), 1.2 ), W ) )
Hz16 = array( map( lambda w: H( A,B,exp(i*w), 1.6 ), W ) )
Hz08 = array( map( lambda w: H( A,B,exp(i*w), 0.8 ), W ) )
Hz12a = array( map( lambda w: H( A,B,exp(i*w), 1.2, f[1] ), W ) )

Hz    = array([H( A,B,exp(i*w), 1   ) for w in W ])
Hz12  = array([H( A,B,exp(i*w), 1.2 ) for w in W ])
Hz16  = array([H( A,B,exp(i*w), 1.6 ) for w in W ])
Hz08  = array([H( A,B,exp(i*w), 0.8 ) for w in W ])
Hz12a = array([H( A,B,exp(i*w), 1.2, f[1] ) for w in W ])


plot(W, abs(Hz[:,N//2]), 'k', linewidth=2)
plot(W, abs(Hz12[:,N//2]), 'k')
plot(W, abs(Hz16[:,N//2]), 'k')
plot(W, abs(Hz08[:,N//2]), 'k')

show()
