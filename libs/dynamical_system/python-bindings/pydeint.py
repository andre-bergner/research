import scipy.weave as weave
import numpy as np
from  operator_overloading  import *

x , y , z = states(3)
a = Parameter(1)
b = Parameter(2)
c = Parameter(3)

"""
a.set( 0.15 )
b.set( 0.2 )
c.set( 8.5 )

dxdt = [
  -y - z,
  x + a*y,
  b + z*(x-c)
]
"""

dxdt = [
  -y - z,
  x + 0.15*y,
  0.2 + z*(x-8.5)
]



system_def_proto = \
"""
   typedef  std::tr1::array< double , %dim% >  State;

   void  System( const State & x , State & dxdt , double t ) {
%dxdt%
   }


   template< class BlitzArray >
   class  WriteIntoBlitzArray {
      int          _index;
      BlitzArray & _array;
   public:
      WriteIntoBlitzArray( BlitzArray & array , size_t index )
      : _array( array ) , _index( index )
      { }

      void operator() ( const State & x , const double t ) {
         for ( int n = 0 ; n < x.size() ; ++n )
            _array(_index,n) = x[n];
         ++_index;
      }
   };

   template< class BlitzArray >
   WriteIntoBlitzArray< BlitzArray >
   create_WriteIntoBlitzArray( BlitzArray & array , size_t index ) {
      return WriteIntoBlitzArray<BlitzArray>( array , index );
   }
"""

dxdt_str = ""
for n in xrange(len(dxdt)):
   dxdt_str +=  "      dxdt[" + str(n) + "] = " + dxdt[n].eval() + ";\n"


system_def = system_def_proto.replace( "%dim%" , str( len(dxdt) ) )
system_def = system_def.replace( "%dxdt%" , dxdt_str )


main_code = \
"""
   State _x = {{ x(0,0) , x(0,1) , x(0,2) }};
   boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<State>() ,
      System , _x , 0.0 , 0.1 , N , create_WriteIntoBlitzArray( x , 1 )
   );
"""

#x = np.array([ 1. , 0. ])

N = 1000
x = np.zeros((N,3))
x[0] = [ 1. , 1. , 1. ]


weave.inline(
   main_code,
   [ 'x' , 'N' ],
   type_converters = weave.converters.blitz,
   support_code = system_def,
#   force = True,    # needed if support_code changed --> TODO check for change
   headers = [ '<tr1/array>' , '<boost/numeric/odeint.hpp>' ],
   include_dirs = [ '/opt/local/include' , '/Users/endboss/libraries/odeint-v2' ],
   extra_compile_args = [ '-O3' ]
)



#integrate( dxdt , initials = [1,1] , times = [0,10,0.1] )
