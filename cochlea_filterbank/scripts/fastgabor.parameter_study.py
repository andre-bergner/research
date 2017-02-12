#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)

N = 64


f  =  arange(N) * 2*pi/N
r  =  1.0
k  =  0.506

e = exp( 1j*f )
a = r*e
b = ones(N)

I = eye(N)
A = diag(a)
A_ = diag( 1./a )
B = b

D = diag(ones(N-1),1) + diag(ones(N-1),-1)
D[-1,0] = 1
D[0,-1] = 1

P  = dot( A , I + k*D )    # propagator matrix

P_ = diag(diag(P)) + diag(diag(P,-1),1) + diag(diag(P,-1),-1)
P_[-1,0] = P[0,-1]
P_[0,-1] = P[0,-1]
#P = P_

def compute_norm():

   M = A_ + I + k*D
   d0 = diag( M , 0 )
   d1 = diag( M , 1 )
   e  = ones( len(M) )

   a = d0[0]
   b = d1[0]
   c = e[0]
   d = e[0]
   
   for n in range( 1 , N-1 ) :
      b = -d[n] * b / a
      d = e[n] - d * d1[n] / a
      c = ...
      a = d0[n] - d1[n]**2 / a
   
   # final
   a = a - d1[n] * b / a


"""   


def solve_cyc_tridiag( diag, offdiag , b ):

   sum = 0.0
   N = len( diag )
   x = []

   if  N == 1 :
      x.append( b[0] / diag[0] )
      return x


   alpha = [ diag[0] ]
   gamma = [ offdiag[0] / alpha[0] ]
   delta = [ offdiag[-1] / alpha[0] ]

   for i in range( 1 , N-2 ):
      a = diag[i] - offdiag[i-1] * gamma[i-1]
      alpha.append( a )
      gamma.append( offdiag[i] / a )
      delta.append( -delta[-1] * offdiag[i-1] / a

   sum = np.sum( alpha * delta**2 )

   alpha[-2] = diag[-2] - offdiag[-3] * gamma[-3]
   gamma[-2] = (offdiag[-2] - offdiag[-3] * delta[-3]) / alpha[-2]

   alpha[-1] = diag[-1] - sum - alpha[-2] * gamma[-2] * gamma[-2]

   #  update
   z = [ b[0] ]
   for i in range( 1 , N-1 ):
      z.append( b[i] - z[i-1] * gamma[i-1] )

   sum = np.sum( delta * z )


   z[-1] = b[-1] - sum - gamma[-2] * z[-2];
   c = []
   for i in range( 0 , N ):
      c[i] = z[i] / alpha[i];

   #  backsubstitution
   x = zeros( len(c) )
   x[-1] = c[-1]
   x[-2] = c[-2] - gamma[-2] * x[-1]

   if N >= 3:
          for (i = N - 3, j = 0; j <= N - 3; j++, i--)
            {
              x[x_stride * i] = c[i] - gamma[i] * x[x_stride * (i + 1)] - delta[i] * x[x_stride * (N - 1)];
            }
        }
    }

}

"""











def H( z ):    # state space transfer function
   return  linalg.solve( dot( A , z*A_ + I + k*D ) , B )
#   return  linalg.solve( z*I + P , B )


W = linspace(-pi,pi,2000)
Hz = array( map( lambda w: H(exp(1.j*w)) , W ) )

print sort( abs(eig(P)[0]) )


#E = []
#K = linspace( 0 , 2 , 2000 )
#for k in K:
#   E.append( sort( abs(eig( dot( A , I + k*D ) )[0]) )[0] )

#figure()
#plot( K , E , 'k' )


figure()
plot( W , abs(Hz)[:,:] ,'k' )

figure()
plot( abs(Hz[:,N/2]) , 'k' )

show()

