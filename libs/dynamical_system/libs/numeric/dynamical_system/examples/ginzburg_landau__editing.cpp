#  include  <iostream>
#  include  <boost/numeric/odeint.hpp>
#  include  <boost/numeric/dynamical_system.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::dynamical_system;



struct StuartLandauOsc : public static_ode<2> {

   double  a , b , c;

   void  operator () ( const state_type & x , state_type & dxdt , double t ) const {

      double  a_  =  a  -  x[0] * x[0]  -  x[1] * x[1];

      dxdt[0]  =  a_ * x[0]  -  b  * x[1];
      dxdt[1]  =  b  * x[0]  +  a_ * x[1];
   }

};



template < class State >
void print_state( const State & state ) {
   for ( auto s1 = state.begin() ; s1 != state.end() ; ++s1 ) {
//      cout << (*s1)[0] << " \t" <<  (*s1)[1] << "\t";
      cout << (*s1)[0] << "\t";
   }
   cout << endl;
}


main() {

   typedef   ode_network< StuartLandauOsc , diffusive_coupling >   GinzburgLandau;

   GinzburgLandau       gl ( 100 );

   runge_kutta4< GinzburgLandau::state_type >   stepper;

   double b = 0.589;
   for ( auto vi = vertices(gl).first ; vi != vertices(gl).second ; ++vi ) {
      gl[*vi].a = 1.0;
      gl[*vi].b = 0.9 + 0.2 * (b *= 3.9*(1.-b));
      gl[*vi].state[0] = 1.0;
      gl[*vi].state[1] = 0.0;
   }


//   StuartLandauOsc osc[100];
   StuartLandauOsc::state_type  y[100];
   runge_kutta4< StuartLandauOsc::state_type >   stepper2;

   for ( int n = 0 ; n < 100 ; ++n ) {
      y[n][0] = 1.0;
      y[n][1] = 0.0;

   }


/*
   auto
      vi = vertices(gl).first ,
      vj = ++vi;

   while ( vj != --vertices(gl).second ) {
      add_link ( gl , *vi , *vj ).weight = 6.;
      add_link ( gl , *vj , *vi ).weight = 6.;
      ++vi;
      ++vj;
   }
*/

   gl[0].a  = 1.0;
   auto vi = vertices(gl).first;
   auto vd = *vi;
//   static_cast<StuartLandauOsc*>(vd)->a = 1.0;

   for ( int n=0; n < 2000 ; ++n ) {
      stepper.do_step( gl , gl.state() , 0.0 , 0.05 );
      print_state( gl.state() );

//      for ( size_t m=0 ; m < 100; ++m ) {
//         stepper2.do_step( gl[m] , y[m] , 0.0 , 0.05 );
//         cout << y[m][0] << "\t";
//      }
      cout << endl;
      
/*
   for ( int m=0 ; m < 100; ++m ) {
         stepper2.do_step( osc[m] , y[m] , 0.0 , 0.05 );
         cout << y[m][0] << "\t";
      }
      cout << endl;
*/
   }

}
