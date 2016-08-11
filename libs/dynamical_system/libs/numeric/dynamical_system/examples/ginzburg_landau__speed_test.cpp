#  include  <iostream>
#  include  <list>
#  include  <numeric>
#  include  <boost/numeric/odeint.hpp>
#  include  <boost/numeric/dynamical_system.hpp>
#  include  <boost/ref.hpp>

#  include  "timer.hpp"

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
double energy( const State & state ) {
   double  E = 0.0;
   for ( auto s = state.begin() ; s != state.end() ; ++s )
      for ( auto i = s->begin() ; i != s->end() ; ++i )
         E += *i * *i;

   return E;
}




#  define   NODES    1000
#  define   STEPS    1000




template < class Graph >
void filling_state_direct( Graph & g ) {

   Timer t( "filling direct" );

   for ( int r = 0 ; r < 100000 ; ++r ) {
      for ( auto vi = vertices(g).first ; vi != vertices(g).second ; ++vi ) {
         g[*vi].state[0] = 1.0;
         g[*vi].state[1] = 0.0;
      }
   }      
}   



template < class Graph >
void filling_state_with_StateProxy( Graph & g ) {

   Timer t( "filling using StateProxy" );

   for ( int r = 0 ; r < 100000 ; ++r ) {
      auto  s = g.state();
      for ( auto si = s.begin() ; si != s.end() ; ++si ) {
         (*si)[0] = 1.0;
         (*si)[1] = 0.0;
      }
   }
}

template< class T >
class IterOverList {

   T & _t;

public:
   
   IterOverList( T & t ) : _t( t )   { }

   template < class InState , class OutState >
   inline void  operator() ( const InState&  x , OutState &  dxdt , const double  t ) const
   {
      typename  InState::const_iterator     i = x.begin();
      typename  OutState::iterator          j = dxdt.begin();

      for ( auto n = _t.begin();  n != _t.end();  ++n ) {
         (*n)( *i , *j , t );
         ++i , ++j;
      }
   }

};


template < class T >
IterOverList<T>  iter_over_list ( T & t )   { return  IterOverList<T>( t ); }


/*
void  iter_over_graph ( const GinzburgLandau::state_type&  x , GinzburgLandau::state_type &  dxdt , const double  t ) {

   GinzburgLandau::state_type::const_iterator     i = x.begin();
   GinzburgLandau::state_type::iterator           j = dxdt.begin();
   boost::graph_traits<GinzburgLandau>::vertex_iterator     node;

   for ( node  = vertices(gl).first;  node != vertices(gl).second;  ++node ) {
      gl[*node]( *i , *j , t );
      ++i , ++j;
   }
}
*/




main() {

   typedef
   ode_network< StuartLandauOsc , diffusive_coupling >
   GinzburgLandau;

   GinzburgLandau    gl ( NODES );

   runge_kutta4< GinzburgLandau::state_type >   stepper;

   Timer  *t;


   filling_state_direct( gl );
   filling_state_with_StateProxy( gl );

   cout << endl << endl;

   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

   t = new Timer( "using StateProxy" );
   for ( int n=0; n < STEPS ; ++n )
      stepper.do_step( boost::ref(gl) , gl.state() , 0.0 , 0.05 );
   delete t;
   cout << "Energy: " << energy( gl.state() ) << endl;


   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   
   GinzburgLandau::state_type   x ( NODES );

   for ( auto xi = x.begin() ; xi != x.end() ; ++xi ) {
      (*xi)[0] = 1.0;
      (*xi)[1] = 0.0;
   }

   t = new Timer( "using vector-state" );
   for ( int n=0; n < STEPS ; ++n )
      stepper.do_step( boost::ref(gl) , x , 0.0 , 0.05 );
   delete t;
   cout << "Energy: " << energy( x ) << endl;
   
   
   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

   typedef  std::vector< StuartLandauOsc::state_type >   sl_state;
   sl_state   y ( NODES );

   for ( auto yi = y.begin() ; yi != y.end() ; ++yi ) {
      (*yi)[0] = 1.0;
      (*yi)[1] = 0.0;
   }

   std::list<StuartLandauOsc>   sl ( NODES );

   runge_kutta4< sl_state >   sl_stepper;

   t = new Timer( "using list of systems" );
   for ( int n=0; n < STEPS ; ++n )
      stepper.do_step( iter_over_list( sl ) , y , 0.0 , 0.05 );
   delete t;
   cout << "Energy: " << energy( x ) << endl;

   
/*
   for ( auto vi = vertices(gl).first ; vi != vertices(gl).second ; ++vi ) {
      gl[*vi].state[0] = 1.0;
      gl[*vi].state[1] = 0.0;
   }


   t = new Timer( "using direct access" );
   for ( int n=0; n < STEPS ; ++n ) {
      int k = 0;
      for ( auto vi = vertices(gl).first ; vi != vertices(gl).second ; ++vi )
         stepper2[k++].do_step( gl[*vi] , gl[*vi].state , 0.0 , 0.05 );
   }
   delete t;
   cout << energy( gl.state() ) << endl;
*/
}


