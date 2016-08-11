from pylab import *


def phase_and_freq ( x1 , x2 ):

   p = arctan2( x2 , x1 )
   w = diff( unwrap( p ) )    # !!! not scaled by time step
   return  p , w



def kuramoto_parameter ( x , y ):

#   px = phase_and_freq( x )
#   py = phase_and_freq( y )
   return  mean( exp( 1.j * (y-x) ) )



def cxcor ( x , y ):

   return  dot ( x , conjugate(y)  ) / sqrt ( dot(x,conjugate(x)) * dot(y,conjugate(y)) )



def loc_set ( x , y , box ):

   dim     =  min( x.shape )
   strobo  =  []
   n       =  0

   for _x in x:
      if  sum( _x > box[0] ) + sum( _x < box[1] )  ==  2*dim :
         strobo.append ( y[n] )
      n += 1

   return  array( strobo )



def area( x ):

   x_min = np.min( x , axis=0 )  # assuming columns
   x_max = np.max( x , axis=0 )

   N  = max( 2 , int( sqrt( len(x) ) / 2. ) )   # the number of grid cells for the area binning

#   x_ofs = (x_max - x_min) / N
#   A = 0
#   for n in arange( 0 , 1.0 , 0.25 ):
#      for m in arange( 0 , 1.0 , 0.25 ):
#         A_ = zeros((N,N))
#         idx = array( (N-1)*(x + [n,m]*x_ofs - x_min)/(x_max - x_min) , dtype=int )
#         iy = array(map( lambda _: int(_) , (N-1)*(y + m*y_ofs - y_min)/(y_max - y_min) ))
#         A_[ix,iy] = 1
#         A = A + float(sum(A_)) / (N**2)
#   return (x_max - x_min)*(y_max - y_min)  * A / 16.

   # FIXME the following code is for 2D only

   A = zeros((N,N))
   idx = array( N*0.999999*(x - x_min)/(x_max - x_min) , dtype=int )
   A[ idx[:,0] , idx[:,1] ] = 1

   return (x_max[0] - x_min[0])*(x_max[1] - x_min[1]) * float(sum(A)) / (N**2)

