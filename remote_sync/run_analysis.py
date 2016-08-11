# -*- coding: utf-8 -*-

import subprocess
from StringIO import StringIO
import time
from pylab import *
from scipy.signal import hilbert
import sync_analysis_tools as sat
import multiprocessing


N_k = 100
N_a = 100
A   = arange( N_a+1 ) / float(N_a) * 20.5 + .5
#... the coupling will be choosen in dependence of alpha ....  K   = arange( N_k+1 ) / float(N_k)

R     = []
R_int = []
R_ext = []


def the_task( ka ):

   k = ka[0]
   a = ka[1]

   print k,a

   #_________ data length: 133072 = 2^17 + 2000 (for transients) ___________________
#   proc = subprocess.Popen ( """stuart_landau_hub_motif %f 133073  ....
#   proc = subprocess.Popen ( """stuart_landau.hub_motif.linux -n 5 -c %f -N 110000 -w 3.56789 -a %f""" % (k,a) ,
   proc = subprocess.Popen ( """a.out -n 5 -c %f -N 31000 -w 3.56789 -a %f""" % (k,a) ,
                             shell=True, stdout=subprocess.PIPE )
   stdout, stderr  = proc.communicate();
   x = loadtxt( StringIO( stdout ) )

   x = x[10000:,:]       # throw away transients

   r  = zeros((5,5))*1.j

   phase = []
   freq  = []
   for n in range( 5 ):
      px,wx =  sat.phase_and_freq ( x[:,2*n] , x[:,2*n+1] )
      phase.append( px )
      freq .append( wx )

   for n in range( 5 ):
      for m in range( n+1 , 5 ):
         r[m,n] = r[n,m] = sat.kuramoto_parameter( phase[n] , phase[m] )

   return  r



def get_results( rr ):

   #_______  summation for star topology __________________
   def sum_int( M ):  return  sum( abs( M[1:,0]  ) ) / 4.
   def sum_ext( M ):  return  sum( abs( M[1:,1:] ) ) / 12.

   #_______  summation for chain topology _________________
#   def sum_int( M ):  return  sum( abs( M[1:4,1:4] ) ) / 6.
#   def sum_ext( M ):  return  abs( M[4,0] )

   for r in rr:
      R.append( r )
      R_int.append( sum_int(r) )
      R_ext.append( sum_ext(r) )


KA = []
for a in A:
#   for k in linspace( 0 , a , N_k+1 ):
   for k in linspace( 0 , 1.5 , N_k+1 ):
      KA.append( (k,a) )

pool = multiprocessing.Pool ( None )
res = pool.map_async( the_task , KA , callback = get_results )
res.wait()     # Wait on the results

R_ext  =  reshape ( array(R_ext) , ( N_k+1 , N_a+1 ) )
R_int  =  reshape ( array(R_int) , ( N_k+1 , N_a+1 ) )

savetxt( "results.stuart_landau.rext(k,a).txt" , R_ext )
savetxt( "results.stuart_landau.rint(k,a).txt" , R_int )

