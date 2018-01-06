#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N  = 100
dw = 2 * pi / N

k1   =  3. * dw
k2   =  0.
damp =  6. * dw
#k    =  4. * dw
#damp =  -2. * dw

# implicit version
#k2 = -1.7*k1
#k1 = 0



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

M1 = dot( A , I + k1*K )
M2 = I + k2*K


def H ( A , B , z  ):
   return  linalg.solve( z * M2  +  M1  , B )

i = 1.j
N_W = 4000
W = linspace(-pi,pi,N_W)

Hz = array([H(A, B, exp(i*w)) for w in W])

S = linalg.solve(M2, M1)
print(sort(abs(eig(S)[0])))

figure()
plot(W, abs(Hz)[:,:],'k', linewidth = 0.7, alpha = 0.6)
plot(W, abs(Hz[:,N//2]), 'k', linewidth = 3)

figure()
plot(abs(np.sum((exp(-24j*f) * Hz)[:,N//2:], axis = 1 )))

figure()
plot( -diff(unwrap(angle(Hz[:,N//2]))) * N_W / (2*pi), 'k' )
plot( abs(Hz[:,N//2]), 'b' )

figure()
plot( abs(fft(Hz[:,10]))[:-100:-1], 'k', linewidth = 2 )
plot( real(fft(Hz[:,10]))[:-100:-1], 'k', alpha = 0.8 )
plot( imag(fft(Hz[:,10]))[:-100:-1], 'k', alpha = 0.8 )

show()
