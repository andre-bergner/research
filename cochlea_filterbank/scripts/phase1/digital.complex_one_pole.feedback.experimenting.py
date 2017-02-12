from pylab import *
np.set_printoptions(linewidth=140)

N = 32

f = arange( N ) * 2*pi/N
damp = .5

d1 = -0.1*exp( -2j*f ) # 0. - 2.0j
d2 = -0.3 + 0.5j

d1 = -0.4 - 0.3j
d2 = +0.5 - 0.5j


a = 0.9*exp( 1j*f + damp )
b = ones(N)

I = eye(len(a))
A = diag(a)
B = b

D = diff(diff(eye(N+1)).T)

#twist = -.4j
#diag( exp(twist) * diag(D,-1),-1) + diag(diag(D)) + diag( exp(twist) * diag(D,1),1)

D[-1,0] = D[-2,-1]
D[0,-1] = D[1,0]


D = diff( eye(N+1) )[:-1]
D[0,-1] = 1


Bk  = I + d1*D
Fw  = dot(A,I + d2*D.T)   # forward matrix


def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * Bk  +  Fw  , B )


i = 1.j
W = linspace(-pi,pi,2000)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )


S = linalg.solve( Bk , Fw )

print sort( abs(eig(S)[0]) )


figure()
plot ( W,abs(Hz)[:,:] ,'k' )
#plot ( W,abs( dot(D,Hz.T).T )[:,:] ,'k' )
#semilogy ( abs(Hz)[:,1:-1] ,'k' )

figure()
plot( abs(Hz[:,N/2]) , 'k' )

show();

