#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
import desolver

np.set_printoptions(linewidth=140)

N    =  64      #  64
damp = -0.12    # -0.12
k    =  0.5     #  0.5
I    = eye(N)

f = arange( N ) * 2*pi/N

a = exp( damp + 1j*f )
e = exp( -1j*f )

A = diag(a)
B = (1. - a*e)
b = (1 + 0.002*rand(N))


D = diff(diff(eye(N+1)).T)
D -= diag(diag(D))
D[-1,0] = D[-2,-1]
D[0,-1] = D[1,0]

K = diag(ones(N-1),1) + diag(ones(N-1),-1)
K[-1,0] = 1
K[0,-1] = 1


M  = dot( A , I + k*K )


def FilterBankTransfer ( A , B , z ):    # state space transfer function
   return  N.linalg.solve( z * I  +  M  , B )





class MySolver ( desolver.DESolver ):

   def error_func ( self, indiv, *args ):

      import numpy as np

      N    =  64      #  64
      damp = -0.12    # -0.12
      k    =  0.5     #  0.5
      I    = np.eye(N)

      f = args[0]
      W = args[1]
      A = args[2]
      B = args[3] * np.exp( -indiv )
      
      K = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
      K[-1,0] = 1
      K[0,-1] = 1

      M  = np.dot( A , I + k*K )


      error = 0

      def FilterBankTransfer ( z ):    # state space transfer function
         return  np.linalg.solve( z * I  +  M  , B )

      Hz = np.array( map( lambda w: FilterBankTransfer( np.exp(1.j*w) ) , W ) )

      max_H = np.max( abs(Hz) , axis = 0 )
      error +=  sum( (1 - max_H)**2 )
      
#      error += death panalety 

      return error







if __name__ == '__main__':

   W  = linspace( -pi , pi , 1000 )

   solver = MySolver(                      # set up the solver
      param_ranges = [ ( 0. , 10. ) ] * len(b),
      population_size = 80,
      max_generations = 300,
      method = desolver.DE_BEST_1,
      args = ( f , W , A , B ),
      scale=0.9, crossover_prob=0.9,
      goal_error = .01,
      polish=True, verbose=True,
      use_pp = True,
      pp_depfuncs = [],
      pp_modules=['numpy']
   )

   print "Best generation:", solver.best_generation
   print "Best individual:", solver.best_individual
   print "Best error:", solver.best_error


   B *= solver.best_individual
   Hz = array( map( lambda w: FilterBankTransfer(A,B,exp(i*w)) , W ) )


   S = linalg.solve( I , M )
   print sort( abs(eig(S)[0]) )

   figure()
   plot( W , abs(Hz)[:,:] ,'k' )

   figure()
   plot( abs(Hz[:,1*N/4]) , 'k' )
   plot( abs(Hz[:,2*N/4]) , 'k' )
   plot( abs(Hz[:,3*N/4]) , 'k' )

   show()
