import scipy.weave as weave
import numpy as np

import operator_overloading

system_def = \
"""
   typedef  std::tr1::array< double , 2 >  State;

   void  System( const State & x , State & dxdt , double t ) {
      dxdt[0] =  x[1] - 0.1*x[0];
      dxdt[1] = -x[0] - 0.1*x[1];
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

main_code = \
"""
   State _x = {{ x(0,0) , x(0,1) }};
   boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<State>() ,
      System , _x , 0.0 , 0.1 , N , create_WriteIntoBlitzArray( x , 1 )
   );
"""

#x = np.array([ 1. , 0. ])

N = 1000
x = np.zeros((N,2))
x[0] = [ 1. , 0. ]


weave.inline(
   main_code,
   [ 'x' , 'N' ],
   type_converters = weave.converters.blitz,
   support_code = system_def,
#   force = True,    # needed if support_code changed --> TODO check for change
   headers = [ '<tr1/array>' , '<boost/numeric/odeint.hpp>' ],
   include_dirs = [ '/opt/local/include' , '/Users/endboss/work/odeint-v2' ],
   extra_compile_args = [ '-O3' ]
)
