#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

#include <tclap/CmdLine.h>

#include <boost/numeric/odeint.hpp>
#include <boost/range/irange.hpp>

#include <boost/numeric/dynamical_system/coupling.hpp>
#include <boost/numeric/dynamical_system/ode_network.hpp>


//#  define      __COMPUTE_LYAPUNOV_SPECTRUM__


////////////////////////////////////////////////////////////////////
//
//  DEFINITION OF SYSTEMS
//

using boost::numeric::odeint::ode;       // FIXME custom namespace
using boost::numeric::odeint::state_t;   // FIXME custom namespace
using boost::numeric::odeint::vtx;       // FIXME custom namespace


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
      t  +=  std::inner_product( first1->begin() , first1->end() , first2->begin() , 0.0);
      ++first1;
      ++first2;
   }
   return t;
}


template <class sys_state, class lyap_type>
void gram_schmidt( sys_state& U , lyap_type& lyap )
{
   size_t  N = U[0].size() * U[0][0].size();    // # of nodes * dimension of single system

   double norm;

   norm = sqrt( my_inner_product( U[0].begin(), U[0].end(), U[0].begin() , 0.0 ) );
   normalize( U[0].begin() , U[0].end() , 1./norm );
   lyap[0]  +=  log ( norm );

   for ( size_t n=1 ; n < N ; ++n ) {

      auto v = U[n];   // assumming const state type

      for ( size_t m=0 ; m < n ; ++m ) {
         double  ip = my_inner_product( v.begin(), v.end(), U[m].begin() , 0.0 );
         subtract_vector( U[n].begin() , U[n].end() , U[m].begin() , ip );
      }

      norm = sqrt( my_inner_product( U[n].begin(), U[n].end(), U[n].begin() , 0.0 ) );
      normalize( U[n].begin() , U[n].end() , 1./norm );
      lyap[n]  +=  log ( norm );
   }
}






/////////////////////////////////////////////////////////////////////////////////
//
//  graph models



//////////    o   o    /////////////////////
//////////     \ /     /////////////////////
//////////      O      /////////////////////
//////////     / \     /////////////////////
//////////    o   o    /////////////////////

template <class Graph>
void star( Graph& g, int node_num, double w_0, double w_width )
{
   g.clear();

   for ( auto n : boost::irange(1,node_num) )
   {
      add_link( g , vtx(n) <= vtx(0) ).weight = 1.0;
      add_link( g , vtx(0) <= vtx(n) ).weight = 1.0 / double(node_num-1);
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

template <class Graph>
void twin_star( Graph& g, int node_num )
{
   double w = 2.5;

//   add_link( g , vtx(0) <= vtx(0) ).weight =  -1.0;
//   add_link( g , vtx(1) <= vtx(1) ).weight =  -1.0;
   add_link( g , vtx(0) <= vtx(1) ).weight =  1.0 / double( node_num/2 );
   add_link( g , vtx(1) <= vtx(0) ).weight =  1.0 / double( node_num/2 );
   g[0].set_w( 2.50 );
   g[1].set_w( 2.51 );
   for ( size_t n = 2 ; n < node_num ; ++n ) {
      int  c  =  n & 1;
      add_link( g , vtx(n) <= vtx(c) ).weight =  1.0;
//      add_link( g , vtx(n) <= vtx(n) ).weight = -1.0;
//      add_link( g , vtx(c) <= vtx(n) ).weight =  1.0 / double( (node_num-2)/2 + 1 - c );
      add_link( g , vtx(c) <= vtx(n) ).weight =  1.0 / double( node_num/2 );
      g[n].set_w( 1.0 + 0.005*n );
   }
}




//////////               /////////////////////
//////////   O-o-o-o-O   /////////////////////
//////////               /////////////////////
template <class Graph>
void chain( Graph& g, int node_num )
{
   size_t n = 0;
   add_link( g , vtx(n) <= vtx(n+1) ).weight =  1.0;
   add_link( g , vtx(n) <= vtx(n)   ).weight = -1.0;
//   g[n].set_w( sqrt(8.) );
   g[n].set_w( 2.5 );
   for ( ++n ; n < node_num-1 ; ++n ) {
      add_link( g , vtx(n) <= vtx(n+1) ).weight =  1.0;
      add_link( g , vtx(n) <= vtx(n-1) ).weight =  1.0;
      add_link( g , vtx(n) <= vtx(n)   ).weight = -2.0;
      g[n].set_w( 1.0 + 0.01*n );
   }
   add_link( g , vtx(n) <= vtx(n-1) ).weight =  1.0;
   add_link( g , vtx(n) <= vtx(n)   ).weight = -1.0;
   g[n].set_w( 2.52 );
}







namespace std
{
   template <typename Iter>  auto begin( pair<Iter,Iter> p )  { return p.first; }
   template <typename Iter>  auto end( pair<Iter,Iter> p )    { return p.second; }
}



template <class Graph>
void print_graph ( Graph g )
{
   using namespace std;

   for ( auto const& v : vertices(g) )
   {
      cout << v << " [ " << g[v].a << ", " << g[v].c << ", " << g[v].w << " ]  <-- ";

      for ( auto const& e : in_edges(v,g) )
         cout << source ( e , g ) << " [ " << g[e].weight << " ] , ";

      cout << endl;
   }
}



template <class Graph1, class Graph2>
void copy_topology( Graph1 const& g1, Graph2& g2 )
{
   g2.clear();

   for ( auto const& v : vertices(g1) )
   {
      for ( auto const& e : out_edges(v,g1) )
      {
         typename boost::graph_traits<Graph2>::edge_descriptor new_e;
         tie(new_e, std::ignore) = add_edge( source(e,g1), target(e,g1), g2 );
         g2[new_e].weight = g1[e].weight;
      }
      g2[v].a = g1[v].a;
      g2[v].c = g1[v].c;
      g2[v].w = g1[v].w;
   }
}







template <class Graph>
void read_adj_matrix( std::istream& is , Graph& g )
{
   g.clear();

   std::string  line;
   long    n_target = 0;
   while ( getline(is,line) )
   {
      std::istringstream sline( line );
      double x;
      long   n_source = 0;
      while ( sline >> x )
      {
         if ( x != 0.0 )
         {
//            cout << "adding link: " << n_target << " <-- " << n_source << endl;
            add_link( g , vtx(n_target) <= vtx(n_source) ).weight =  x; }
            ++n_source;
         }
      ++n_target;
   }
}

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


constexpr auto app_version = "0.7";

struct Params
{
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

   std::string
      adj_file = "",
      frq_file = "",
      file_name = "";
};


Params  parse_options(int arg_num , char** args);


int main( int arg_num , char** arg )
{
   using namespace std;
   using namespace boost::numeric::odeint;


   using sys_t = ode_network< node_sys , coupling >;
   using lsys_t = ode_network< lnode_sys, lcoupling >;

   sys_t   sys;
   lsys_t  ls;

   auto p = parse_options( arg_num , arg );

   if ( p.verbose_output )
   cout << "_________________________________________________" << endl
        << "network motif of coupled oscillators, Version " << app_version << endl << endl;


   if ( p.adj_file != "" ) {
//      cout << " reading adjacency matrix from file '" << p.adj_file << "'." << endl;
      ifstream adj_stream( p.adj_file.c_str() );
      read_adj_matrix( adj_stream, sys );
      adj_stream.close();

      if ( p.frq_file != "" ) {
//         cout << " reading oscillator's frequencies from file '" << p.frq_file << "'." << endl;
         ifstream frq_stream( p.frq_file.c_str() );
         double f;
         boost::graph_traits<sys_t>::vertex_iterator v1,v2;
//         sys_t::vertex_iterator v1,v2;
         tie(v1,v2) =  vertices(sys);
         while ( (frq_stream >> f)  &&  (v1 != v2) )
            sys[*(v1++)].set_w( f );
         frq_stream.close();
      }

      p.node_num = num_vertices(sys);


      print_graph( sys );
      cout << "--------------------------" << endl;
      print_graph( ls );
      cout << "--------------------------" << endl;

   }


   if ( p.verbose_output )
   cout
      << "running simulation with the following parameters:" << endl
      << "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯"  << endl
      << "  coupling strength:\t\t"     << p.cpl         << endl
      << "  amplitude parameter:\t\t"   << p.amp         << endl
      << "  freq parameter:\t\t"        << p.w_0         << endl
      << "  freq dist parameter:\t\t"   << p.w_width     << endl
      << "  number of nodes:\t\t"       << p.node_num    << endl
      << "  adjacency matrix file:\t\t" << p.node_num    << endl

      << "  number of samples:\t\t"     << p.N_samples   << endl
      << "  integrator time-step:\t\t"  << p.dt          << endl
      << "  file name:\t\t\t"           << p.file_name   << endl
      << endl;

   
   const int  node_dim  =  node_sys::dimension;
   const int  net_dim   =  node_dim * p.node_num;

   sys_t::state           x( p.node_num );
   vector<lsys_t::state>  u(net_dim);

   vector<double>  lyap( net_dim , 0.0 );

   runge_kutta4< sys_t::state >  stepper;


   for ( int n=0 ; n < net_dim ; ++n )
      u[n].resize( p.node_num );

   for ( size_t n = 0  ;  n < p.node_num  ;  ++n )  {
      for ( int d=0 ; d < node_dim ; ++d )
         x[n][d]  =  rnd();
      for ( int m=0 ; m < net_dim ; ++m )
         for ( int d=0 ; d < node_dim ; ++d )
            u[m][n][d]  =  ( node_dim*n+d == m ) ?  1.0 : 0.0 ;
   }

   if ( !num_vertices(sys) ) {
      star(sys, p.node_num, p.w_0, p.w_width);
   //   twin_star(sys, p.node_num);  twin_star(ls, p.node_num);
   //   chain(sys, p.node_num);      chain(ls, p.node_num);
   }

   copy_topology( sys , ls );

   for ( int n = 0 ; n < p.node_num ; ++n ) {
//      double a = 1.0 + 0.05*rnd();
      sys[n].set_a( p.amp );
      ls[n].set_a( p.amp );
   }


// TODO   this could be solved with a global variable in the coupling function
   for (auto e : edges(sys))  sys[e].weight *= p.cpl;

   for ( lsys_t::edge_iterator  node  = edges(ls).first;
                              node != edges(ls).second;
         ls[*node++].weight *= p.cpl );


   if ( p.verbose_output )  print_graph( sys );


   ostream* pos = &cout;

   if ( p.file_name != "" ) {
      pos = new ofstream( p.file_name.c_str() );
      if ( pos ) {  if ( p.verbose_output )
         cout << "sending output to " << p.file_name << endl;
      }
      else {
         cerr << "Error: cannot open file " << p.file_name << ". Sending to standard out." << endl;
         pos = &cout;
      }
   }


   const int N_realization = 1;

   for ( int realization = 0 ; realization < N_realization ; ++realization ) {

      for ( size_t n = 0  ;  n < p.node_num  ;  ++n )  {
         for ( int d=0 ; d < node_dim ; ++d )
            x[n][d]  =  rnd();
         for ( int m=0 ; m < net_dim ; ++m )
            for ( int d=0 ; d < node_dim ; ++d )
               u[m][n][d]  =  ( node_dim*n+d == m ) ?  1.0 : 0.0 ;
      }


//      int  N_steps_before_renormalization  =  int( 1. / dt );
      int  N_steps_before_renormalization  =  10;
      for ( size_t k = 0 ; k < p.N_samples ; ++k ) {

         stepper.do_step( sys , x , 0.0 , p.dt );     // integrate the system

//         for ( int n=0 ; n < p.node_num ; ++n )
//            x[n][0]  =  fmod( x[n][0] , 2.*pi );    // post process phase to ensure num. stability

#        ifdef    __COMPUTE_LYAPUNOV_SPECTRUM__

            for ( int n=0 ; n < p.node_num ; ++n )
               for ( int d=0; d < node_dim ; ++d )
                  ls[n].set( d , x[n][d] );

            for ( int n=0 ; n < net_dim ; ++n )
               stepper.do_step( ls , u[n] , 0.0 , p.dt );    // integrate the linearized system

            if ( !N_steps_before_renormalization-- ) {
               gram_schmidt( u , lyap );
               N_steps_before_renormalization  =  10;
            }

#        else

           for ( size_t n=0 ; n < p.node_num ; ++n )
              for ( int d=0 ; d < node_dim ; ++d )
                 *pos << x[n][d] << "\t";
           *pos << endl;

#        endif
      }

   }

#  ifdef    __COMPUTE_LYAPUNOV_SPECTRUM__
      for ( size_t n=0 ; n < net_dim ; ++n )
         *pos << lyap[n] / (p.dt * p.N_samples * N_realization) << endl;
#  endif

}












Params parse_options( int arg_num , char** arg )
{
  using namespace std;
  using namespace TCLAP;

  Params p;

  try {
    CmdLine cmd( "cmd_message" , ' ' , app_version );


    //
    // Define arguments
    //

    ValueArg<double>  arg_cpl( "c" , "coupling" , "coupling strength" , false , p.cpl , "double" );
    cmd.add( arg_cpl );


    ValueArg<double>  arg_a( "a" , "amplitude" , "amplitude parameter" , false , p.amp , "double" );
    cmd.add( arg_a );

    ValueArg<double>  arg_w( "w" , "freq" , "frequency of detuned node" , false , p.w_0 , "double" );
    cmd.add( arg_w );

    ValueArg<double>  arg_d( "d" , "freq_dist" , "frequency distribution of bulk nodes" , false , p.w_width , "double" );
    cmd.add( arg_d );

    ValueArg<int>  arg_node_num( "n" , "nodenum" , "number of nodes" , false , p.node_num , "int" );
    cmd.add( arg_node_num );

    ValueArg<string>  arg_adj_file_name(
               "A", "adjacency" , "filename with an adjacency matrix" , false , p.adj_file , "file name" );
    cmd.add( arg_adj_file_name );

    ValueArg<string>  arg_frq_file_name(
               "F", "frequencies" , "filename with an frequencies" , false , p.frq_file , "file name" );
    cmd.add( arg_frq_file_name );



    ValueArg<int>  arg_smp_num( "N" , "smpnum" , "number of samples of generated output" , false , p.N_samples , "int" );
    cmd.add( arg_smp_num );

    ValueArg<double>  arg_dt( "t" , "dt" , "integrator time step" , false , p.dt , "double" );
    cmd.add( arg_dt );



    UnlabeledValueArg<string>  arg_file_name( "filename" , "name for file to write the output in" , false , p.file_name , "file name" );
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
    p.cpl      = arg_cpl.getValue();
    p.amp      = arg_a.getValue();
    p.w_0      = arg_w.getValue();
    p.w_width  = arg_d.getValue();
    p.node_num = arg_node_num.getValue();
    p.adj_file = arg_adj_file_name.getValue();
    p.frq_file = arg_frq_file_name.getValue();

    p.N_samples = arg_smp_num.getValue();
    p.dt        = arg_dt.getValue();
    p.file_name = arg_file_name.getValue();

    if ( arg_verbose.isSet() )  p.verbose_output = true;
  }

  catch ( ArgException& e ) {
    cout << "ERROR: " << e.error() << " " << e.argId() << endl;
  }

  return p;
}



