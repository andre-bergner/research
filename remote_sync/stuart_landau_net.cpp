/*
 *   g++ -O3 -I ../../odeint -I ../../../lib 
 */
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>   // for inner_product

#include <tclap/CmdLine.h>

#include <boost/numeric/odeint.hpp>

#include <boost/numeric/dynamical_system/coupling.hpp>
#include <boost/numeric/dynamical_system/ode_network.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace TCLAP;

using boost::tie;


///////////////////////////////////////////////////////////////////
//
//  the parameters of the program
//

//#  define      __COMPUTE_LYAPUNOV_SPECTRUM__

const char app_version[] = "0.7";

double
   cpl      = 0.5,   // coupling                   [0..max]
   amp      = 1.0,   // amplitude/hopf parameter
   w_0      = 2.5,
   w_width  = 0.05,
   dt       = 0.05;

int
   N_samples = 10000,
   node_num  = 5;  // number of nodes in network


bool    verbose_output = false;

string
   adj_file = "",
   frq_file = "",
   file_name = "";

void  parse_options ( int arg_num , char** arg );



////////////////////////////////////////////////////////////////////
//
//  DEFINITION OF SYSTEMS
//


/////////////////////////////////////////////////////////////////
//
//  STUART-LANDAU MODEL
//

MAKE_ODE_WITH_3PARAM(

   stuart_landau , 2 ,

   a , 1.0 ,
   w , 1.0 ,
   c , 0.0 ,

   double  r = x[0] * x[0]  +  x[1] * x[1];
   dxdt[0]  =  a * x[0]  -  w * x[1]  -  r * (    x[0] - c * x[1]);
   dxdt[1]  =  w * x[0]  +  a * x[1]  -  r * (c * x[0] +     x[1]);
)


class linear_stuart_landau : public ode< 2 >
{
public:
   double  a,w,c,x[2];

public:

   linear_stuart_landau() :
      a    ( 1.0 ),
      w    ( 1.0 ),
      c    ( 0.0 )
   {  x[0] = 1.0;
      x[1] = 0.0;
   }

   void set_a ( double _ ) { a = _; }
   void set_w ( double _ ) { w = _; }
   void set_c ( double _ ) { c = _; }

   void set ( int d , double val ) { x[d] = val; }

   void  operator() ( const state & u , state & dudt , const double t ) const
   {
      double r = x[0]*x[0] + x[1]*x[1];
      dudt[0]  =  a*u[0] - w*u[1] - r*(  u[0] - c*u[1]) - 2.*(  x[0]-c*x[1])*(x[0]*u[0]+x[1]*u[1]);
      dudt[1]  =  w*u[0] + a*u[1] - r*(c*u[0] +   u[1]) - 2.*(c*x[0]+  x[1])*(x[0]*u[0]+x[1]*u[1]);
   }
};



typedef state_t<double,2> state2;

MAKE_COUPLING_WITH_2PARAM(

   stuart_landau_coupling,

   weight      ,  1.0 ,
   weight_imag ,  0.0 ,

   state2 tmp;

//   tmp[0] = weight             * y[0]  -   weight*weight_imag * y[1];
//   tmp[1] = weight*weight_imag * y[0]  +   weight             * y[1];

   tmp[0] = weight             * ( y[0] - x[0] )  -   weight*weight_imag * ( y[1] - x[1] );
   tmp[1] = weight*weight_imag * ( y[0] - x[0] )  +   weight      * ( y[1] - x[1] );

//   tmp[0] = weight      * ( y[0] - x[0] )  -   weight_imag * ( y[1] - x[1] );
//   tmp[1] = weight_imag * ( y[0] - x[0] )  +   weight      * ( y[1] - x[1] );
   return tmp;
)


/////////////////////////////////////////////////////////////////
//
//   KURAMOTO MODEL
//

MAKE_ODE_WITH_1PARAM(

   phase_oscillator , 1 ,

   w , 1.0 ,

   dxdt[0]  =  w;
)


class lin_phase_oscillator : public ode< 1 >
{
   double  w , x[1];

public:

   lin_phase_oscillator() :
      w    ( 1.0 )
   {  x[0] = 1.0;
   }

   void set_w ( double _ ) { w = _; }
   void set ( int d , double val ) { x[d] = val; }

   void  operator() ( const state & u , state & dudt , const double t ) const
   {
      dudt[0]  =  0.0;
   }
};


typedef state_t<double,1> state1;

MAKE_COUPLING_WITH_1PARAM(

   phase_coupling,

   weight , 1.0 ,

   state1 tmp;
   tmp[0] = weight * sin( y[0] - x[0] );
   return tmp;
)


MAKE_COUPLING_WITH_1PARAM(

   lin_phase_coupling,

   weight , 1.0 ,

   state1 tmp;
   tmp[0] = weight * cos( y[0] - x[0] );
   return tmp;
)






////////////////////////////////////////////////////////////
//
//  FitzHugh - Nagumo



MAKE_ODE_WITH_3PARAM(

   FitzHughNagumo , 2 ,

   a , 0.5 ,
   w , 1.0 ,
   c , 0.1 ,

   float b = 0.1;
   dxdt[0]  =  w * ( x[0] * (a - x[0]) * (x[0] - 1.0) - x[1] );  // + I;
   dxdt[1]  =  w * ( b*x[0] - c*x[1] );
)

/*
class linear_FitzHughNagumo : public ode< 2 >
{
public:
   double  a,w,c,x[2];

public:

   linear_stuart_landau() :
      a    ( 1.0 ),
      w    ( 1.0 ),
      c    ( 0.0 )
   {  x[0] = 1.0;
      x[1] = 0.0;
   }

   void set_a ( double _ ) { a = _; }
   void set_w ( double _ ) { w = _; }
   void set_c ( double _ ) { c = _; }

   void set ( int d , double val ) { x[d] = val; }

   void  operator() ( const state & u , state & dudt , const double t ) const
   {
      double r = x[0]*x[0] + x[1]*x[1];
      dudt[0]  =  a*u[0] - w*u[1] - r*(  u[0] - c*u[1]) - 2.*(  x[0]-c*x[1])*(x[0]*u[0]+x[1]*u[1]);
      dudt[1]  =  w*u[0] + a*u[1] - r*(c*u[0] +   u[1]) - 2.*(c*x[0]+  x[1])*(x[0]*u[0]+x[1]*u[1]);
   }
};
*/


typedef state_t<double,2> state2;

MAKE_COUPLING_WITH_1PARAM(

   FitzHughNagumo_coupling,

   weight  ,  1.0 ,

   state2  tmp;

   tmp[0] = weight  * ( y[0] - x[0] );
   tmp[1] = 0.0;
   return tmp;
)


////////////////////////////////////////////////////////////
//
//  Rössler

/*
MAKE_ODE_WITH_3PARAM(

   roessler , 3 ,

   a , 0.15 ,
   b , 0.2 ,
   c , 8.5 ,

   dxdt[0]  =  -x[1] -   x[2];
   dxdt[1]  =   x[0] + a*x[1];
   dxdt[2]  =   b + x[2] * (x[0] - c);
)
*/


class roessler : public ode< 3 >
{
   double  a,b,c,w;
public:
   roessler() :
      a ( 0.15 ),
      b ( 0.2 ),
      c ( 8.5 ),
      w ( 1.0 )
   { }

   void set_a ( double _ ) { a = _; }
   void set_b ( double _ ) { b = _; }
   void set_c ( double _ ) { c = _; }
   void set_w ( double _ ) { w = _; }        // TODO
   
   void  operator() ( const state & x , state & dxdt , const double t ) const
   {
      dxdt[0]  =  -w*x[1] -   x[2];
      dxdt[1]  =   w*x[0] + a*x[1];
      dxdt[2]  =   b + x[2] * (x[0] - c);
   }
};



class linear_roessler : public ode< 3 >
{
   double  a,c,w,x[3];
public:
   linear_roessler() :
      a ( 0.15 ),
      c ( 8.5 ),
      w ( 1.0 )
   {  x[0] = 1.0;
      x[1] = 0.0;
      x[2] = 0.0;
   }

   void set_a ( double _ ) { a = _; }
   void set_b ( double _ ) {        }        // not needed - just for the sake of complete interface
   void set_c ( double _ ) { c = _; }

   void set_w ( double _ ) { w = _; }        // TODO
   
   void set ( int d , double val ) { x[d] = val; }

   void  operator() ( const state & u , state & dudt , const double t ) const
   {
      dudt[0]  =   -w*u[1] -   u[2];
      dudt[1]  =    w*u[0] + a*u[1];
      dudt[2]  = x[2]*u[0] + (x[0]-c)*u[2];
   }
};



typedef state_t<double,3> state3;

MAKE_COUPLING_WITH_1PARAM(

   roessler_coupling,

   weight , 1.0 ,

   state3 tmp;
//   tmp[0] = weight * ( y[0] - x[0] );     // coupling in the x-component
//   tmp[1] = weight * ( y[1] - x[1] );     // coupling in the y-component
   tmp[0] = weight * y[0];     // coupling in the x-component
   tmp[1] = weight * y[1];     // coupling in the y-component
   return tmp;
)





typedef     stuart_landau              node_sys;
typedef     linear_stuart_landau      lnode_sys;
typedef     stuart_landau_coupling     coupling;
typedef     stuart_landau_coupling    lcoupling;
//typedef     diffusive_coupling    coupling;
//typedef     diffusive_coupling   lcoupling;

/*
typedef     FitzHughNagumo            node_sys;
typedef     linear_stuart_landau      lnode_sys;
typedef     FitzHughNagumo_coupling     coupling;
typedef     stuart_landau_coupling    lcoupling;
*/




/*
typedef     roessler              node_sys;
typedef     linear_roessler      lnode_sys;
typedef     roessler_coupling     coupling;
typedef     roessler_coupling    lcoupling;
*/

/*
typedef     phase_oscillator         node_sys;
typedef     lin_phase_oscillator    lnode_sys;
typedef     phase_coupling           coupling;
typedef     lin_phase_coupling      lcoupling;
*/

double  rnd ()   { return  (double)rand() / (double)RAND_MAX; }








template< class iterator , class T >
void normalize( iterator first , iterator last , T norm_inv ) {
   while ( first != last )  *first++ *= norm_inv;
}

template< class iterator , class T >
void  subtract_vector ( iterator first1 , iterator last1 , iterator first2 , T val ) {
   while ( first1 != last1 )   *first1++  -=  val * *first2++;
}


template< class iterator , class T >
T  my_inner_product ( iterator first1 , iterator last1 , iterator first2 , T ) {
   T t = 0.0;
   while ( first1 != last1 ) {
      t  +=  inner_product( first1->begin() , first1->end() , first2->begin() , 0.0);
      ++first1;
      ++first2;
   }
   return t;
}


template < class sys_state , class lyap_type >
void gram_schmidt ( sys_state *U , lyap_type & lyap )
{
   size_t  N = U[0].size() * U[0][0].size();    // # of nodes * dimension of single system

   double norm;

   norm = sqrt( my_inner_product( U[0].begin() , U[0].end() ,U[0].begin() , 0.0 ) );
   normalize( U[0].begin() , U[0].end() , 1./norm );
   lyap[0]  +=  log ( norm );

   for ( size_t n=1 ; n < N ; ++n ) {

      sys_state  v( U[n] );   // assumming const state type

      for ( size_t m=0 ; m < n ; ++m ) {
         double  ip = my_inner_product( v.begin() , v.end() , U[m].begin() , 0.0 );
         subtract_vector( U[n].begin() , U[n].end() , U[m].begin() , ip );
      }

      norm = sqrt( my_inner_product( U[n].begin() , U[n].end() ,U[n].begin() , 0.0 ) );
      normalize( U[n].begin() , U[n].end() , 1./norm );
      lyap[n]  +=  log ( norm );
   }
}


/////////////////////////////////////////////////////////////////////////////////

//////////    o   o    /////////////////////
//////////     \ /     /////////////////////
//////////      O      /////////////////////
//////////     / \     /////////////////////
//////////    o   o    /////////////////////

template< class graph >
void star( graph & g ) {

   g.clear();

   for ( size_t n = 1 ; n < node_num ; ++n ) {
      add_link ( g , _node(n) <= _node(0) ).weight =  1.0;
      add_link ( g , _node(0) <= _node(n) ).weight =  1.0 / double(node_num-1);
//      g[n].set_w( 1.0 + 0.01*n );
//      g[n].set_w( 0.975 + 0.01*n );
      if ( node_num == 2 )
         g[n].set_w( 1.1 );
      else
         g[n].set_w( 1.1 - 0.5*w_width + float(n-1)*w_width/float(node_num-2)  );
//      g[n].set_w( 1.0 );
   }
   g[0].set_w( w_0 );
}




//////////    o   o  o   o   /////////////////////
//////////     \ /    \ /    /////////////////////
//////////      O------O     /////////////////////
//////////     / \    / \    /////////////////////
//////////    o   o  o   o   /////////////////////

template< class graph >
void twin_star( graph & g ) {

   double w = 2.5;

//   add_link ( g , _node(0) <= _node(0) ).weight =  -1.0;
//   add_link ( g , _node(1) <= _node(1) ).weight =  -1.0;
   add_link ( g , _node(0) <= _node(1) ).weight =  1.0 / double( node_num/2 );
   add_link ( g , _node(1) <= _node(0) ).weight =  1.0 / double( node_num/2 );
   g[0].set_w( 2.50 );
   g[1].set_w( 2.51 );
   for ( size_t n = 2 ; n < node_num ; ++n ) {
      int  c  =  n & 1;
      add_link ( g , _node(n) <= _node(c) ).weight =  1.0;
//      add_link ( g , _node(n) <= _node(n) ).weight = -1.0;
//      add_link ( g , _node(c) <= _node(n) ).weight =  1.0 / double( (node_num-2)/2 + 1 - c );
      add_link ( g , _node(c) <= _node(n) ).weight =  1.0 / double( node_num/2 );
      g[n].set_w( 1.0 + 0.005*n );
   }
}




//////////               /////////////////////
//////////   O-o-o-o-O   /////////////////////
//////////               /////////////////////
template< class graph >
void chain( graph & g ) {

   size_t n = 0;
   add_link ( g , _node(n) <= _node(n+1) ).weight =  1.0;
   add_link ( g , _node(n) <= _node(n)   ).weight = -1.0;
//   g[n].set_w( sqrt(8.) );
   g[n].set_w( 2.5 );
   for ( ++n ; n < node_num-1 ; ++n ) {
      add_link ( g , _node(n) <= _node(n+1) ).weight =  1.0;
      add_link ( g , _node(n) <= _node(n-1) ).weight =  1.0;
      add_link ( g , _node(n) <= _node(n)   ).weight = -2.0;
      g[n].set_w( 1.0 + 0.01*n );
   }
   add_link ( g , _node(n) <= _node(n-1) ).weight =  1.0;
   add_link ( g , _node(n) <= _node(n)   ).weight = -1.0;
   g[n].set_w( 2.52 );
}


/*
 //  just a pair of oscillators
   add_link (  s , _node(0) <= _node(1) ).weight =  1.0;
   add_link (  s , _node(1) <= _node(0) ).weight =  1.0;
   add_link ( ls , _node(0) <= _node(1) ).weight =  1.0;
   add_link ( ls , _node(1) <= _node(0) ).weight =  1.0;

   s[1].set_w( 1.0 );
  ls[1].set_w( 1.0 );
*/





template <
   class Graph >
void print_graph ( Graph g ) {

   for ( typename Graph::vertex_iterator
           node  = vertices(g).first;  node != vertices(g).second; ++node ) {

//      cout << *node << " <-- ";
      cout << *node << " [ " << g[*node].a << ", " << g[*node].c << ", " << g[*node].w << " ]  <-- ";

      for ( typename Graph::in_edge_iterator
              link  = in_edges(*node,g).first;  link != in_edges(*node,g).second; ++link )
//         cout << source ( *link , g ) << "  ";
         cout << source ( *link , g ) << " [ " << g[*link].weight << " ] , ";

      cout << endl; }}






template <
   class Graph1,
   class Graph2 >
void copy_topology ( const Graph1 & g1, Graph2 & g2 ) {

   g2.clear();

   typename boost::graph_traits<Graph1>::vertex_iterator  vi, vi_end;

   for ( tie(vi, vi_end) = vertices(g1);  vi != vi_end ;  ++vi) {
      typename boost::graph_traits<Graph1>::out_edge_iterator  ei, ei_end;
      for ( tie(ei, ei_end) = out_edges(*vi, g1);  ei != ei_end ;  ++ei ) {
         typename boost::graph_traits<Graph2>::edge_descriptor new_e;
         bool inserted;
         tie(new_e, inserted) = add_edge( source(*ei, g1), target(*ei, g1), g2 );
         g2[new_e].weight = g1[*ei].weight;
//         cout << g1[*ei].weight  << endl;
         }
      g2[*vi].a = g1[*vi].a;
      g2[*vi].c = g1[*vi].c;
      g2[*vi].w = g1[*vi].w;
      }}







template < class Graph >
void read_adj_matrix( istream & is , Graph & g ) {

   g.clear();

   string  line;
   long    n_target = 0;
   while ( getline(is,line) ) {
      istringstream sline( line );
      double x;
      long   n_source = 0;
      while ( sline >> x ) {
         if ( x != 0.0 ) {
//            cout << "adding link: " << n_target << " <-- " << n_source << endl;
            add_link ( g , _node(n_target) <= _node(n_source) ).weight =  x; }
         ++n_source;
         }
      ++n_target; }}

/*
   string line;
   while ( getline(is,line) ) {
      istringstream sline( line );
      double x;
      while ( sline >> x ) {
         cout << x << ", "; }
      cout << endl; }}
*/



////////////////////////////////////////////////////////////////////////////////////
//
//  MAIN PROGRAM
//


typedef  ode_network < node_sys , coupling >   sys;
typedef  ode_network <lnode_sys ,lcoupling >  lsys;

sys   s;
lsys  ls;


int main( int arg_num , char** arg )
{
   if ( verbose_output )
   cout << "_________________________________________________" << endl
        << "network motif of coupled oscillators, Version " << app_version << endl << endl;

   parse_options( arg_num , arg );

   if ( adj_file != "" ) {
//      cout << " reading adjacency matrix from file '" << adj_file << "'." << endl;
      ifstream adj_stream( adj_file.c_str() );
      read_adj_matrix( adj_stream , s );
      adj_stream.close();

      if ( frq_file != "" ) {
//         cout << " reading oscillator's frequencies from file '" << frq_file << "'." << endl;
         ifstream frq_stream( frq_file.c_str() );
         double f;
         boost::graph_traits<sys>::vertex_iterator v1,v2;
//         sys::vertex_iterator v1,v2;
         tie(v1,v2) =  vertices(s);
         while ( (frq_stream >> f)  &&  (v1 != v2) )
            s[*(v1++)].set_w( f );
         frq_stream.close();
      }

      node_num = num_vertices(s);

/*
      print_graph( s );
      cout << "--------------------------" << endl;
      print_graph( ls );
      cout << "--------------------------" << endl;
*/
   }

   if ( verbose_output )
   cout
      << "running simulation with the following parameters:" << endl
      << "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯"  << endl
      << "  coupling strength:\t\t"     << cpl         << endl
      << "  amplitude parameter:\t\t"   << amp         << endl
      << "  freq parameter:\t\t"        << w_0         << endl
      << "  freq dist parameter:\t\t"   << w_width     << endl
      << "  number of nodes:\t\t"       << node_num    << endl
      << "  adjacency matrix file:\t\t" << node_num    << endl

      << "  number of samples:\t\t"     << N_samples   << endl
      << "  integrator time-step:\t\t"  << dt          << endl
      << "  file name:\t\t\t"           << file_name   << endl
      << endl;

   
   const int  node_dim  =  node_sys::dimension;
   const int  net_dim   =  node_dim * node_num;

//   sys          s( node_num );
//  lsys         ls( node_num );
   sys::state   x( node_num );
   std::vector<lsys::state>  u(net_dim);

   vector< double >  lyap ( net_dim , 0.0 );

   runge_kutta4< sys::state >  stepper;


   for ( int n=0 ; n < net_dim ; ++n )
      u[n].resize( node_num );

   for ( size_t n = 0  ;  n < node_num  ;  ++n )  {
      for ( int d=0 ; d < node_dim ; ++d )
         x[n][d]  =  rnd();
      for ( int m=0 ; m < net_dim ; ++m )
         for ( int d=0 ; d < node_dim ; ++d )
            u[m][n][d]  =  ( node_dim*n+d == m ) ?  1.0 : 0.0 ;
   }

   if ( !num_vertices(s) ) {
      star( s );
   //   twin_star( s );  twin_star( ls );
   //   chain( s );      chain( ls );
   }

   copy_topology( s , ls );


   for ( int n = 0 ; n < node_num ; ++n ) {
//      double a = 1.0 + 0.05*rnd();
      s[n].set_a( amp );
     ls[n].set_a( amp );
   }


// TODO   this could be solved with a global variable in the coupling function
   for ( sys::edge_iterator  node  = edges(s).first;
                             node != edges(s).second;
         s[*node++].weight *= cpl );

   for ( lsys::edge_iterator  node  = edges(ls).first;
                              node != edges(ls).second;
         ls[*node++].weight *= cpl );


   if ( verbose_output )  print_graph( s );


   ostream* pos = &cout;

   if ( file_name != "" ) {
      pos = new ofstream( file_name.c_str() );
      if ( pos ) {  if ( verbose_output )
         cout << "sending output to " << file_name << endl;
      }
      else {
         cerr << "Error: cannot open file " << file_name << ". Sending to standard out." << endl;
         pos = &cout;
      }
   }


   const int N_realization = 1;

   for ( int realization = 0 ; realization < N_realization ; ++realization ) {

      for ( size_t n = 0  ;  n < node_num  ;  ++n )  {
         for ( int d=0 ; d < node_dim ; ++d )
            x[n][d]  =  rnd();
         for ( int m=0 ; m < net_dim ; ++m )
            for ( int d=0 ; d < node_dim ; ++d )
               u[m][n][d]  =  ( node_dim*n+d == m ) ?  1.0 : 0.0 ;
      }


//      int  N_steps_before_renormalization  =  int( 1. / dt );
      int  N_steps_before_renormalization  =  10;
      for ( size_t k = 0 ; k < N_samples ; ++k ) {

         stepper.do_step( s , x , 0.0 , dt );     // integrate the system

//         for ( int n=0 ; n < node_num ; ++n )
//            x[n][0]  =  fmod( x[n][0] , 2.*pi );    // post process phase to ensure num. stability

#        ifdef    __COMPUTE_LYAPUNOV_SPECTRUM__

            for ( int n=0 ; n < node_num ; ++n )
               for ( int d=0; d < node_dim ; ++d )
                  ls[n].set( d , x[n][d] );

            for ( int n=0 ; n < net_dim ; ++n )
               stepper.do_step( ls , u[n] , 0.0 , dt );    // integrate the linearized system

            if ( !N_steps_before_renormalization-- ) {
               gram_schmidt ( u , lyap );
               N_steps_before_renormalization  =  10;
            }

#        else

           for ( size_t n=0 ; n < node_num ; ++n )
              for ( int d=0 ; d < node_dim ; ++d )
                 *pos << x[n][d] << "\t";
           *pos << endl;

#        endif
      }

   }

#  ifdef    __COMPUTE_LYAPUNOV_SPECTRUM__
      for ( size_t n=0 ; n < net_dim ; ++n )
         *pos << lyap[n] / (dt*N_samples*N_realization) << endl;
#  endif

}












void parse_options( int arg_num , char** arg )
{
  try {
    CmdLine cmd( "cmd_message" , ' ' , app_version );

    //
    // Define arguments
    //

    ValueArg<double>  arg_cpl( "c" , "coupling" , "coupling strength" , false , cpl , "double" );
    cmd.add( arg_cpl );


    ValueArg<double>  arg_a( "a" , "amplitude" , "amplitude parameter" , false , amp , "double" );
    cmd.add( arg_a );

    ValueArg<double>  arg_w( "w" , "freq" , "frequency of detuned node" , false , w_0 , "double" );
    cmd.add( arg_w );

    ValueArg<double>  arg_d( "d" , "freq_dist" , "frequency distribution of bulk nodes" , false , w_width , "double" );
    cmd.add( arg_d );

    ValueArg<int>  arg_node_num( "n" , "nodenum" , "number of nodes" , false , node_num , "int" );
    cmd.add( arg_node_num );

    ValueArg<string>  arg_adj_file_name(
               "A", "adjacency" , "filename with an adjacency matrix" , false , adj_file , "file name" );
    cmd.add( arg_adj_file_name );

    ValueArg<string>  arg_frq_file_name(
               "F", "frequencies" , "filename with an frequencies" , false , frq_file , "file name" );
    cmd.add( arg_frq_file_name );



    ValueArg<int>  arg_smp_num( "N" , "smpnum" , "number of samples of generated output" , false , N_samples , "int" );
    cmd.add( arg_smp_num );

    ValueArg<double>  arg_dt( "t" , "dt" , "integrator time step" , false , dt , "double" );
    cmd.add( arg_dt );



    UnlabeledValueArg<string>  arg_file_name( "filename" , "name for file to write the output in" , false , file_name , "file name" );
    cmd.add( arg_file_name );

    SwitchArg  arg_verbose ( "v" , "verbose" , "verbose output" , false );
    cmd.add( arg_verbose );


    //
    // Parse the command line.
    //
    cmd.parse( arg_num , arg );


    //
    // get the values
    //
    cpl      = arg_cpl.getValue();
    amp      = arg_a.getValue();
    w_0      = arg_w.getValue();
    w_width  = arg_d.getValue();
    node_num = arg_node_num.getValue();
    adj_file = arg_adj_file_name.getValue();
    frq_file = arg_frq_file_name.getValue();

    N_samples = arg_smp_num.getValue();
    dt        = arg_dt.getValue();
    file_name = arg_file_name.getValue();

    if ( arg_verbose.isSet() )  verbose_output = true;
  }

  catch ( ArgException& e ) {
    cout << "ERROR: " << e.error() << " " << e.argId() << endl;
  }
}



