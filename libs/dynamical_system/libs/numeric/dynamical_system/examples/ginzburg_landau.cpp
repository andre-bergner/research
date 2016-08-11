#  include  <iostream>
#  include  <boost/numeric/odeint.hpp>
#  include  <boost/numeric/dynamical_system.hpp>
#  include  <boost/ref.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::dynamical_system;



struct StuartLandauOsc : public static_ode<2> {

   double  a , b , c;

   void  operator () ( const state_type & x , state_type & dxdt , double t ) const {

      double  a_  =  a  -  x[0] * x[0]  -  x[1] * x[1];     // minus ????

      dxdt[0]  =  a_ * x[0]  -  b  * x[1];
      dxdt[1]  =  b  * x[0]  +  a_ * x[1];
   }

};



template < class State >
void print_state( const State & state , double t = 0.) {
   for ( auto s1 = state.begin() ; s1 != state.end() ; ++s1 ) {
//      cout << (*s1)[0] << " \t" <<  (*s1)[1] << "\t";
      cout << (*s1)[0] << "\t";
   }
   cout << endl;
}



#  define   NODES    100
#  define   STEPS    1000


main() {

   typedef
      ode_network< StuartLandauOsc , diffusive_coupling >
      GinzburgLandau;

   GinzburgLandau    gl ( NODES );

   runge_kutta4< GinzburgLandau::state_type >   stepper;

   GinzburgLandau::state_type   x ( NODES );

   double b = 0.589;
   for ( auto vi = vertices(gl).first ; vi != vertices(gl).second ; ++vi ) {
      gl[*vi].a = 1.0;
      gl[*vi].b = 0.9 + 0.2 * (b *= 3.9*(1.-b));
      gl[*vi].state[0] = 1.0;
      gl[*vi].state[1] = 0.0;
   }

   for ( auto n = x.begin(); n != x.end(); ++n ) {
      (*n)[0] = 1.0;
      (*n)[1] = 0.0;
   }

   auto
      vi = vertices(gl).first ,
      vj = vi;

   while ( ++vj != vertices(gl).second ) {
      add_link( gl , *vi , *vj ).weight = .2;
      add_link( gl , *vj , *vi ).weight = .2;
      ++vi;
   }


   for ( int n=0; n < STEPS ; ++n ) {
//      stepper.do_step( boost::ref( gl ) , gl.state() , 0.0 , 0.05 );
      stepper.do_step( boost::ref( gl ) , x , 0.0 , 0.05 );
//      print_state( gl.state() );
      print_state( x );
   }

   integrate_n_steps(
      runge_kutta4<GinzburgLandau::state_type>(),
      boost::ref(gl) , x , 0. , 0.05 , 100 ,
      print_state<GinzburgLandau::state_type>
   );
}




