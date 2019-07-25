#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__  = "André Bergner"
__version__ = "0.2"

import numpy as np
import math
from math import sqrt, log , pow, exp, ceil


def closest_anti_prime( n ):
   """
   computes the to 'n' closest integer 'm' > 'n', which can be factored into the primes 2,5,7.
   Will be used by cwt in order to speed up the fft, which is fastes if the data length
   can be factored into many small primes.
   """

   l2 = log( 2.0 )
   l3 = log( 3.0 )
   l5 = log( 5.0 )
   ln = log(  n  )

   x_max  =  ceil( ln / l2 )
   m      =  pow( 2.0 , x_max )	# first guess

   for  x in range( 0 , int(x_max) + 1 ):

      y_max  =  math.ceil( (ln - l2*x) / l3 )

      for y in range( 0 , int(y_max) + 1 ):

         z  =  ceil( ( ln - l2*x - l3*y ) / l5 )
         m_ =  pow( 2.0 , x ) * pow( 3.0 , y ) * pow( 5.0 , z )
         if  m_ < m  :  m = m_

   return  int ( m )


#  _________________________________________________________
#  now come some standard wavelets

#class Wavelet

#class cauchy:
#   """
#   Cauchy-Paul Wavelet
#   """

#   def __call__ ( w , Q = 2. ):
#      q  =  1./Q;  
#      a  =  ( sqrt(3.)*sqrt(768.*q + 6912.) - 4.*q + 144.) /  ((9.*sqrt(32.) + 54.)*q)
#      return  np.exp( a*( np.log(w) - w + 1. ) )


def cauchy( w , Q = 2. ):
   """
   Cauchy-Paul Wavelet
   """
   q  =  1./Q;  
   a  =  ( sqrt(3.)*sqrt(768.*q + 6912.) - 4.*q + 144.) /  ((9.*sqrt(32.) + 54.)*q)
   a = Q
   return np.exp( a*( np.log(w) - w + 1. ) )


# alternative name for cauchy wavelet
def paul( w , Q = 2. ):
   """
   Cauchy-Paul Wavelet
   """
   return cauchy( w , Q )


def morlet( w , Q = 2. ):
   """
   Morlet Wavelet
   """
   a  =  2.0 * Q * log(2.0)	# Q-version: Q = w_max / (w_hi - w_lo)
   return np.exp( -a*(w-1.)**2 )


def log_morlet( w , Q = 5 ):
   """
   Log-Morlet Wavelet -- has a gaussian shaped kernel on a logarithmic scale/frequency axis
   """
   return np.exp( -(Q*np.log(w))**2 )


def derivative( N ):
   tmp1 = np.array( range(0,N/2) )
   tmp2 = np.array( range(-N/2,0) )
   return   2.j*math.pi*np.concatenate(( tmp1 , tmp2 )) / float(N)



def cwt(
   x ,
#   frequencies = np.exp(np.arange( -5.5 , 0.0 , 0.01 )) ,
   frequencies = np.exp(np.arange( -2.5 , 0.0 , 0.01 )) ,
   wavelet = cauchy,
   Q = 10.
):
   """
   Computes a continuous wavelet transform

   @param x : input data
   @type  x : array of real or complex type

   @param frequencies : Frequencies/Scales at which the CWT is computed, normalized to 1 = Nyquist frequency
   @type  frequencies : array of reals

   @param wavelet : Wavelet
   @type  wavelet : callback function to wavelet

   Examples:
   cwt( data ) -- CWT of data with default parameters
   cwt( data , np.arange( 0.01 , 1.0 , 0.)
   """


   N_x    =  len(x)
   N_pad  =  closest_anti_prime( N_x + 120 ) - N_x
   N      =  N_x + N_pad        # data length including padding

   X = np.fft.fft( np.concatenate(( x , np.zeros(N_pad) )) )	# fft of padded input data
   w = np.arange( 0 , N/2 ) * 2./N 
   # TODO check if frequency scaling is correct ( either Nyquist or zero included or both ? )

   WT = [] 	# the resulting transform


   for f in frequencies:
      a = 1.0 / f
      WT.append( np.fft.ifft( np.concatenate((X[:N/2] * wavelet(a*w,Q) , np.zeros(N/2))) )[:N_x] )   # <-- this makes real w'lets progressive, FIXME

   return  [ np.array(WT) , frequencies ]
   # TODO make this a class behaving like the actual transform with freq and wlet as memebers



def cwt_d(
   x ,
   frequencies = np.exp(np.arange( -5.5 , 0.0 , 0.01 )) ,
   wavelet = cauchy,
   Q = 10.
):
   """ Computes a continuous wavelet transform and its derivative """

   N_x    =  len(x)
   N_pad  =  closest_anti_prime( N_x + 120 ) - N_x
   N      =  N_x + N_pad        # data length including padding
   X = np.fft.fft( np.concatenate(( x , np.zeros(N_pad) )) )	# fft of padded input data
   w = np.arange( 0 , N/2 ) * 2./N 
   WT  = [] 	# the resulting transform
   WTd = [] 	# the resulting transform

   for f in frequencies:
      a = 1.0 / f
      WX  = X[:N/2] * wavelet(a*w,Q)
      WXd = WX * derivative(N)[0:N/2]
      WT.append ( np.fft.ifft( np.concatenate(( WX , np.zeros(N/2))) )[:N_x] )
      WTd.append( np.fft.ifft( np.concatenate(( WXd, np.zeros(N/2))) )[:N_x] )

   return  ( np.array(WT) , np.array(WTd) , frequencies )








def gabor(
   x ,
   frequencies  =  np.arange ( 0.01 , 1.0 , 0.01 ) ,
   Q            =  10.
):
   """
   Computes the Gabor transform

   @param x : input data
   @type  x : array of real or complex type

   @param frequencies : Frequencies/Scales at which the CWT is computed, normalized to 1 = Nyquist frequency
   @type  frequencies : array of reals

   @param Q : quality factor / bandwidth
   @type  Q : float

   Examples:
   gabor( data ) -- CWT of data with default parameters
   gabor( data , np.arange( 0.01 , 1.0 , 0.01) , 3.3 )
   """

   def gab ( w , Q ):
      a  =  2.0 * Q * log(2.0)	# Q-version: Q = w_max / (w_hi - w_lo)
      return np.exp( -a*w**2 )


   N_x    =  len(x)
   N_pad  =  closest_anti_prime( N_x + 120 ) - N_x
   N      =  N_x + N_pad        # data length including padding

   X = np.fft.fft( np.concatenate(( x , np.zeros(N_pad) )) )	# fft of padded input data
   w = np.arange( 0 , N/2 ) * 2./N 
   # TODO check if frequency scaling is correct ( either Nyquist or zero included or both ? )

   GT = [] 	# the resulting transform

   ww = np.concatenate(( w , -w[::-1] ))
   for f in frequencies:
      GT.append( np.fft.ifft( X * gab(ww-f,Q) )[:N_x] )
#      GT.append( np.fft.ifft( np.concatenate((X[:N/2] * gab(w-f,Q) , np.zeros(N/2))) )[:N_x] )   # <-- this makes real w'lets progressive, FIXME

   return  ( np.array(GT) , frequencies )
   # TODO make this a class behaving like the actual transform with freq and wlet as memebers





from copy import deepcopy


def icwt(             # TODO include padded zeros in inverse transorm
   WT_ ,               # as returned by cwt
   wavelet = cauchy,
   Q = 10.
):                    # FIXME how to compute scale measure for summation ?

   WT = deepcopy( WT_ )

   N_x   = len( WT[0][0,:] )
   N_pad = closest_anti_prime( N_x + 120 ) - N_x
   N     = N_x + N_pad		# data length including padding

   w = np.arange( 0 , N/2 ) * 2./N 

   for n in range(len(WT[1])):   # TODO use some form of struct: w.frequencies:
      a  =  1.0 / WT[1][n]
      Vox  =  np.fft.fft( np.concatenate(( WT[0][n,:] , np.zeros(N_pad) )) )
      WT[0][n,:] = np.fft.ifft( np.concatenate((Vox[:N/2] * wavelet(a*w,Q) , np.zeros(N/2))) )[:N_x]

   return  np.sum( WT[0] , axis=0 )



def icwtest(             # TODO include padded zeros in inverse transorm
   WT_ ,               # as returned by cwt
   wavelet = cauchy,
   Q = 10.
):                    # FIXME how to compute scale measure for summation ?

   WT = deepcopy( WT_ )

   N_x   = len( WT[0][0,:] )
   N_pad = closest_anti_prime( N_x + 120 ) - N_x
   N     = N_x + N_pad		# data length including padding

   w = np.arange( 0 , N/2 ) * 2./N 

   for n in range(len(WT[1])):   # TODO use some form of struct: w.frequencies:
      a  =  1.0 / WT[1][n]
      Vox  =  np.fft.fft( np.concatenate(( WT[0][n,:] , np.zeros(N_pad) )) )
      WT[0][n,:] = np.fft.ifft( np.concatenate((Vox[:N/2] * wavelet(a*w,Q) , np.zeros(N/2))) )[:N_x]

   return  WT



# _________________________________________________________________
# now some tools



import pylab as mp

def plotcwt( x , wavelet = cauchy , Q = 10. , fignum = None ):

   W = cwt( x , wavelet = wavelet , Q = Q )[0]

   mp.figure( num = fignum , figsize = (16,8) )

   a1 = mp.axes([0.1, 0.3, 0.8, 0.7])
   ratio = (float(W.shape[1]) / float(W.shape[0]))
   mp.imshow( abs(W) , aspect = 0.4*ratio , origin='lower')

   a2 = mp.axes([0.1, 0.1, 0.8, 0.2])
   mp.plot( x , 'k' )
   mp.xlim([ 0 , len(x) ])

   mp.show()

   return W



def usage():
   print """
____________________________________________________________________
cwt.py """ + __version__ + """, written by """ + __author__ + """

Usage:  cwt.py [options] filename [filename2]

Options:
  -q    sets the wavelet's quality factor (ration of center-frequency to bandwidth)
  -w    chooses a wavelet

Examples:
  - just use standard parameters

       cwt.py data.txt

  - perform a cwt with the Morlet wavelet and a sharp frequency resolution:

       cwt.py -q 30 -w morlet data.txt
"""




def main():
   import  sys , getopt as go

   try:
      opts , args  =  go.getopt ( sys.argv[1:], "hq:w:", ["help"])

   except  go.GetoptError, err:
      # print help information and exit:
      print str(err) # will print something like "option -a not recognized"
      usage()
      sys.exit(2)

   if len( args ) == 0:
      print "Error: No file specified!"
      usage()
      sys.exit(1)


   # set the standard parameters
   wavelets = {
      "cauchy" : cauchy,
      "paul"   : paul,
      "morlet" : morlet,
      "derivative" : derivative
   }
   wavelet  =  "cauchy"
   Q        =  5.

   for  opt , arg  in  opts:
      if    opt in ( "-h", '--help')  :  usage()
      elif  opt == '-q'               :  Q = float(arg)
      elif  opt == '-w'               :  wavelet = arg
   # FIXME catch error if wrong wlet name is specified

   print "q:", Q
   print "wavelet: " + wavelet

   try:  x = np.loadtxt( args[0] )
   except IOError:
      print "error: could not open file '" + args[0] + "'"
      sys.exit( -1 )


   np.fft.ifft(plotcwt( x , wavelet = wavelets[wavelet] , Q = Q ))



# import cProfile

if __name__ == '__main__':
   main()
#    cProfile.run( "main()" )

