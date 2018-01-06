#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

i = 1.j
N  = 100
N  = 200
dw = 2 * pi / N

k1   =  3. * dw
damp =  6. * dw
damp =  4. * dw
#k    =  4. * dw
#damp =  -2. * dw

# implicit coupling:
#k2 = -1.7*k1
#k1 = 0
k2 = 0
rescale = -0.93
rescale = -0.8


def ptridiag( left , middle , right ):
    matrix = diag( left[:-1] , -1)  +  diag( middle )  +  diag( right[1:] , +1 )
    matrix[-1,0] = left[0]
    matrix[0,-1] = right[-1]
    return matrix


f = arange(N) * dw
a = exp(1j*f + damp)
b = ones(N)

I = eye(len(a))
A = diag(a)
B = b / N

K = ptridiag(ones(N), -2*ones(N), ones(N))
#K = ptridiag( ones(N) , zeros(N) , ones(N) )
#phi = exp( 5*dw*1.j )
#K = ptridiag( phi*ones(N) , -2*ones(N) , conj(phi)*ones(N) )

M1 = dot(A, I + k1*K)
M2 = I + k2*K


#M2 = I - rescale*M1
#P = diag(exp( 1j*f ))

def H(A, B, z, s=0):    # state space transfer function
    ## return  linalg.solve( z*I + P*s  +  dot( I + P*s*z , M1 )  , dot( I + P*s*z , B) )
    return  linalg.solve(z * (s*M1 + M2) + (M1 + s*M2), B*(1 + s*z))

def freq_warp( z , s ):
    return  (s + z) / (1 + s*z)

i = 1.j
N_W = 4000
W = linspace(-pi,pi,N_W)

#Hz = array( map( lambda w: H(A,B, freq_warp( exp(i*w), -0.85 )) , W ) )
Hz = array([H(A, B, exp(i*w), rescale) for w in W])


S = linalg.solve(M2, M1)
print(sort( abs(eig(S)[0])))
print("-------------------------------------------------------------------")

S = linalg.solve( rescale*M1 + M2 , M1 + rescale*M2 )
print(sort( abs(eig(S)[0])))


def pole_zero_plot(poles, zeros=None):
    phase = linspace(0, 2*pi, 1000)
    plot(cos(phase), sin(phase), 'k')
    scatter(real(poles) , imag(poles), color='k', marker='x')
    title("Poles (x) and Zeros (o)")
    xlim([ -1.2 , 1.2 ])
    ylim([ -1.2 , 1.2 ])


def set_labels():
    xticks( [-pi,0,pi] , [r'$-\pi$' , 0 , r'$\pi$'])
    xlim((-pi , pi ))
    xlabel(r"$\omega$")
    ylabel(r"$A(\omega)$")

figure()
plot(W, abs(Hz)[:,:], 'k', linewidth=0.7, alpha=0.6)
plot(W, abs(Hz[:,N//2]), 'k', linewidth=3)
phase_factor = exp(-20j * f)
plot(W, abs( np.sum((phase_factor * Hz)[:,N//2:], axis=1)))
set_labels()
title("amplitude response for all channels")

figure()
for phase in linspace(-50, 0, 10):
    phase_factor = exp(phase * 1j * f)
    plot(W, abs(np.sum((phase_factor * Hz)[:,N//2:], axis=1)), label="{}".format(phase))
    plot(W, -unwrap(angle(np.sum((phase_factor * Hz)[:,N//2:], axis=1))), label="{}".format(phase))
set_labels()
legend()
title("reconstructions")

figure()
plot(W[:-1], -diff(unwrap(angle(Hz[:,N//2]))) * N_W / (2*pi), 'k')
plot(W, abs(Hz[:,N//2]), 'b')
set_labels()

figure()
plot(abs(fft(Hz[:,10]))[:-100:-1], 'k', linewidth=2)
plot(real(fft(Hz[:,10]))[:-100:-1], 'k', alpha=0.8)
plot(imag(fft(Hz[:,10]))[:-100:-1], 'k', alpha=0.8)

figure()
pole_zero_plot(-eig(S)[0])

show();

