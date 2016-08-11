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






def area2D_inline( x ):

   from instant import inline_with_numpy

   c_code = """
   double  area (
      int Nx_rows, int Nx_cols, double* x,
      int N
   ){
      double
         x0_min  =  100000000000000.0,
         x0_max  = -100000000000000.0,
         x1_min  =  100000000000000.0,
         x1_max  = -100000000000000.0;

      for ( int n=0 ; n < Nx_rows; ++n ) {
         if ( x[n*Nx_cols+0] > x0_max )   x0_max = x[n*Nx_cols+0];
         if ( x[n*Nx_cols+0] < x0_min )   x0_min = x[n*Nx_cols+0];
         if ( x[n*Nx_cols+1] > x1_max )   x1_max = x[n*Nx_cols+1];
         if ( x[n*Nx_cols+1] < x1_min )   x1_min = x[n*Nx_cols+1];
      }
      
      double  x0_scale  =  .999999 / (x0_max - x0_min);
      double  x1_scale  =  .999999 / (x1_max - x1_min);

      int *A  =  new int[ N * N ];
      memset ( A , 0 , sizeof(int) * N * N );

      for ( int k=0 ; k < Nx_rows; ++k ) {
         int id_x0  =  int( double(N) * x0_scale * (x[k*Nx_cols+0] - x0_min) );
         int id_x1  =  int( double(N) * x1_scale * (x[k*Nx_cols+1] - x1_min) );
         A[ id_x0*N + id_x1 ] = 1;
      }

      int A_sum = 0;
      for ( int n=0 ; n < N*N; ++n )   A_sum  +=  A[n];

      delete A;

      return  (x0_max-x0_min)*(x1_max-x1_min) * double( A_sum ) / double(N*N);
   }
   """

   N  = max( 2 , int( sqrt( len(x) ) / 2. ) )   # the number of grid cells for the area binning

   c_area = inline_with_numpy ( c_code, arrays = [ ['Nx_rows','Nx_cols','x'] ])

   return  c_area( x , N )


