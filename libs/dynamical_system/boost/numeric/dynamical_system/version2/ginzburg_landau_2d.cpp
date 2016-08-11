/* Boost numeric/odeint/examples/ginzburg_landau.cpp
 
  Copyright 2009 André Bergner

  Shows the usage of odeint and the dynamical_system-framework
  in order to integrate the discretized Ginzburg-Landau-Equation:

    du/dt  =  (a+ib)u – c u|u|² + Δu,
    u is complex

  Distributed under the Boost Software License, Version 1.0.
  (See accompanying file LICENSE_1_0.txt or
  copy at http://www.boost.org/LICENSE_1_0.txt)
*/


#include <cstdlib>
#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/dynamical_system/additive_coupleable.hpp>
#include <boost/numeric/dynamical_system/const_dynamical_network.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::dynamical_system;


const size_t	NODE_NUM  = 16;
const double	dt        = 0.05;
const size_t	N_samples = 5000;


//----- using a macro for a quick definition -----------------
MAKE_ODE_WITH_3PARAM(

   general_hopf , 2 ,

   a , 1.0 ,
   b , 1.0 ,
   c , 0.0 ,

   double  r = x[0] * x[0]  +  x[1] * x[1];
   dxdt[0]  =  a * x[0]  -  b * x[1]  -  r * (    x[0] - c * x[1]);
   dxdt[1]  =  b * x[0]  +  a * x[1]  -  r * (c * x[0] +     x[1]);
)

double  rnd ()  { return  (double)rand() / (double)RAND_MAX; }

int main( int argc , char **argv )
{
   typedef  const_dynamical_network <
               additive_coupleable < const_dynamical_network <
                  additive_coupleable < general_hopf >,
                  NODE_NUM > >,
               NODE_NUM
            > sys;

   sys          s;
   sys::state   x;
   stepper_rk4 < sys::state >  integrator;

   for ( size_t m=0 ; m < NODE_NUM ; ++m ) {
      for ( size_t n=0 ; n < NODE_NUM ; ++n ) {
         x[m][n][0] = rnd();
         x[m][n][1] = rnd();
         s[m][n].set_a ( rnd() );
         s[m][n].set_b ( 0.6 + 0.2*rnd() );
         s[m][n].set_c ( 0.4 + rnd() );
      }
      s[m].set_1D_open_laplacian();
   }

   s.set_1D_open_laplacian();
//   s.set_coupling(1.0);

   double t = 0.0;
   for( size_t k=0 ; k < N_samples ; ++k , t += dt ) {
      integrator.do_step( s , x , t , dt );
      for ( size_t m=0 ; m < NODE_NUM ; ++m )
         for ( size_t n=0 ; n < NODE_NUM ; cout << x[m][n++][0] << "\t" );
      cout << endl;
   }

   return 0;
}


/*
  Compile with
  g++ -O3 -Wall -I$BOOST_ROOT -I../../../../ ginzburg_landau_constant_step.cpp 
*/

