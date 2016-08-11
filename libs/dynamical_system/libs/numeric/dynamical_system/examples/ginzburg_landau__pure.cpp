#  include  <iostream>
#  include  <vector>
#  include  <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;




struct GinzburgLandau {

   typedef  vector<double>   state_type;

   double            weight;
   vector<double>    a , b , c;


   GinzburgLandau ( size_t N )
   :  a ( N ),
      b ( N ),
      c ( N )
   { }

   void  operator () ( const state_type & x , state_type & dxdt , double t ) const {

      size_t   N = a.size();

      for ( size_t n=0; n < N ; ++n )
         _one_node( x , dxdt , n );

      dxdt[0] += weight * ( x[2] - x[0] );
      dxdt[1] += weight * ( x[3] - x[1] );

      for ( size_t n=2; n < 2*N-2 ; ++n )
         dxdt[n] += weight * ( x[n+2] + x[n-2] - 2.*x[n] );

      dxdt[2*N-2] += weight * ( x[2*N-4] - x[2*N-2] );
      dxdt[2*N-1] += weight * ( x[2*N-3] - x[2*N-1] );
   }

private:

   void  _one_node ( const state_type & x , state_type & dxdt , size_t n ) const
   {
      double  a_  =  a[n]  -  x[2*n] * x[2*n]  -  x[2*n+1] * x[2*n+1];
      
      dxdt[2*n  ]  =  a_   * x[2*n]  -  b[n] * x[2*n+1];
      dxdt[2*n+1]  =  b[n] * x[2*n]  +  a_   * x[2*n+1];
   }

};



template < class State >
void print_state( const State & state ) {
   for ( auto s1 = state.begin() ; s1 != state.end() ; s1+=2 ) {
//      cout << (*s1)[0] << " \t" <<  (*s1)[1] << "\t";
      cout << *s1 << "\t";
   }
   cout << endl;
}



#  define   NODES    100
#  define   STEPS    1000


main() {

   GinzburgLandau              gl (  NODES  );
   GinzburgLandau::state_type  x  ( 2*NODES );

   runge_kutta4< GinzburgLandau::state_type >   stepper;

   double b = 0.589;
   for ( size_t n = 0 ; n < NODES ; ++n ) {
      gl.a[n] = 1.0;
      gl.b[n] = 0.9 + 0.2 * (b *= 3.9*(1.-b));
      x[2*n  ] = 1.0;
      x[2*n+1] = 0.0;
   }
   gl.weight = 0.2;

   for ( int n=0; n < STEPS ; ++n ) {
      stepper.do_step( gl , x , 0.0 , 0.05 );
      print_state( x );
   }

}
