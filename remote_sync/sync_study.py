# -*- coding: utf-8 -*-

import subprocess
import time
from pylab import *
from scipy.signal import hilbert
import sync_analysis_tools as sat


try:
    from StringIO import StringIO      # python 2
    def make_IO(stream) : return StringIO(stream)

except ImportError:
    from io import StringIO            # python 3
    def make_IO(stream) : return BytesIO(stream)

from io import BytesIO



N = 100
K = arange(N+1) / float(N)

R = []
R_int = []
R_ext = []
R_all = []

C = []
C_int = []
C_ext = []
C_all = []

R2 = []
R2_int = []
R2_ext = []
R2_all = []

L = []
L_int = []
L_ext = []
L_all = []

A = []
A_int = []
A_ext = []
A_all = []

t0   = time.time()
runs = 0

observer_square = [[0.,-.2] , [100000.,.2]]


for k in K:

   print(k)

   t1 = time.time()
   if runs:
      time_per_run  =  (t1-t0) / float(runs)
      rest_time  =  time_per_run * float(N+1) - (t1-t0)
      print("remaining time: " + str(rest_time))

   #_________ data length: 133072 = 2^17 + 2000 (for transients) ___________________ # 65536 + 2000
   proc = subprocess.Popen ( """remote_sync -c %f -N 67536""" % k , shell=True, stdout=subprocess.PIPE )
#   proc = subprocess.Popen ( """net_motif %f 4048""" % k , shell=True, stdout=subprocess.PIPE )
   stdout, stderr  = proc.communicate();
   x = loadtxt( make_IO( stdout ) )

   x = x[2000:,:]       # throw away transients

   r  = zeros((5,5))*1.j
   c  = zeros((5,5))*1.j
   r2 = zeros((5,5))*1.j
   l  = zeros((5,5))
   axp= zeros((5,5))    # area of cross phases

   #print "analyzing..."

   A_att = []
   phase = []
   freq  = []
   hfreq = []
   for n in range(5):
      px,wx =  sat.phase_and_freq ( x[:,2*n] , x[:,2*n+1] )
      phase.append( px )
      freq .append( wx )
      hfreq.append( hilbert(wx-mean(wx)) )
      #A_att.append( sat.area( x[:,2*n:2*n+2] ) )

   for n in range(5):
      for m in range(n+1, 5):
         r[m,n] = r[n,m] = sat.kuramoto_parameter( phase[n] , phase[m] )
         c[m,n] = c[n,m] = sat.cxcor( x[:,2*n] + 1.j*x[:,2*n+1] , x[:,2*m] + 1.j*x[:,2*m+1])
         r2[m,n]= r2[n,m]= sat.kuramoto_parameter( angle(hfreq[n]) , angle(hfreq[m]) )

         # Tiago's stroboscobic method based on localizes sets
         # s = sat.loc_set ( x[:,2*m:2*m+2] , x[:,2*n:2*n+2] , observer_square )
         # A_obs = sat.area( s )
         # if A_obs > A_att[n] :  A_obs = A_att[n]     # in case of 1D-attractor we have bad sampling
         # l[m,n] = l[n,m] = 1.  -  A_obs / A_att[n]

         #axp[m,n] = axp[n,m] = sat.area(
         #                         ascontiguousarray( array([phase[n],phase[m]]).T ) ) / (2.*pi)**2


   #  summation for star topology
   def sum_int(M):  return  sum( abs( M[1:,0]  ) ) / 4.
   def sum_ext(M):  return  sum( abs( M[1:,1:] ) ) / 12.

   #  summation for chain topology
#   def sum_int( M ):  return  sum( abs( M[1:4,1:4] ) ) / 6.
#   def sum_ext( M ):  return  abs( M[4,0] )

   R.append( r )
   R_int.append( sum_int(r) )
   R_ext.append( sum_ext(r) )
   R_all.append( sum( abs( r ) ) / 20 )

   C.append( c )
   C_int.append( sum_int(c) )
   C_ext.append( sum_ext(c) )
   C_all.append( sum( abs( c ) ) / 20 )

   R2.append( r2 )
   R2_int.append( sum_int(r2)  )
   R2_ext.append( sum_ext(r2) )
   R2_all.append( sum( abs( r2 ) ) / 20 )

   L.append( l )
   L_int.append( sum_int(l) )
   L_ext.append( sum_ext(l) )
   L_all.append( sum( abs( l ) ) / 20 )

   #A.append( l )
   #A_int.append( sum_int(axp) )
   #A_ext.append( sum_ext(axp) )
   #A_all.append( sum( abs( axp ) ) / 20 )

   runs += 1




savetxt(
   "results.txt",
   array([ K , R_int , R_ext , R_all,
               C_int , C_ext , C_all,
               R2_int, R2_ext, R2_all,
               L_int , L_ext , L_all,
               #A_int , A_ext , A_all
        ]).T )

import matplotlib.font_manager
leg_prop = matplotlib.font_manager.FontProperties( size = 16 )


figure(1)
legend.fontsize = 25
plot ( K , R_ext , label = r"$r_{ext}$" )
plot ( K , R_int , label = r"$r_{int}$" )
plot ( K , R_all , label = r"$r_{all}$" )
legend( prop = leg_prop )
xlabel( r"$k$" , fontsize=22 )
ylabel( r"$r$" , fontsize=22 )


figure(2)
legend.fontsize = 25
plot ( K , C_ext , label = r"$c_{ext}$" )
plot ( K , C_int , label = r"$c_{int}$" )
plot ( K , C_all , label = r"$c_{all}$" )
legend( prop = leg_prop )
xlabel( r"$k$" , fontsize=22 )
ylabel( r"$c$" , fontsize=22 )


figure(3)
legend.fontsize = 25
plot ( K , R2_ext , label = r"$r2_{ext}$" )
plot ( K , R2_int , label = r"$r2_{int}$" )
plot ( K , R2_all , label = r"$r2_{all}$" )
legend( prop = leg_prop )
xlabel( r"$k$" , fontsize=22 )
ylabel( r"$r2$" , fontsize=22 )


figure(4)
legend.fontsize = 25
plot ( K , L_ext , label = r"$l_{ext}$" )
plot ( K , L_int , label = r"$l_{int}$" )
plot ( K , L_all , label = r"$l_{all}$" )
legend( prop = leg_prop )
xlabel( r"$k$" , fontsize=22 )
ylabel( r"$l$" , fontsize=22 )

#figure(5)
#legend.fontsize = 25
#plot ( K , A_ext , label = r"$Axp_{ext}$" )
#plot ( K , A_int , label = r"$Axp_{int}$" )
#plot ( K , A_all , label = r"$Axp_{all}$" )
#legend( prop = leg_prop )
#xlabel( r"$k$" , fontsize=22 )
#ylabel( r"$Axp$" , fontsize=22 )

#show()
show(block=True)
