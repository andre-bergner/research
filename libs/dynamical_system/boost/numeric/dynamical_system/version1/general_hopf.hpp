/*
 boost header: numeric/dynamical_system/general_hopf.hpp

 Copyright 2009 Andre Bergner


 This class provides a generalized Hopf normal form, a differential equation like this

   z = (a+ib)z - (1+ic) |z|^2 z
   z complex  with  z = (u,v)

  parameter:
    a  -  Hopf bifurcation control parameter
    b  -  eigen-frequency of autonomous system
    c  -  additional parameter coupling the phase and the amplitude


 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
*/


#ifndef		BOOST_NUMERIC_DYNAMICAL_SYSTEMS_GENERAL_HOPF
#define		BOOST_NUMERIC_DYNAMICAL_SYSTEMS_GENERAL_HOPF

#include   <boost/numeric/dynamical_system/const_dynamical_system.hpp>


namespace boost {
namespace numeric {
namespace dynamical_system {


  class general_hopf : public const_dynamical_system< 2 >
  {
    double
      a,   // the hopf parameter
      b,   // the eigen-frequency
      c;   // additional imaginary part, is used e.g. in complex Ginzburg-Landau-Eq.

  public:

    void  operator() ( const state & x , state & dxdt , const double t ) const
    {
      double  r  =  x[0] * x[0]  +  x[1] * x[1];
      dxdt[0]  =  a * x[0]  -  b * x[1]  -  r * (    x[0] - c * x[1]);
      dxdt[1]  =  b * x[0]  +  a * x[1]  -  r * (c * x[0] +     x[1]);
    }

    // std & non-std  c'tor
    general_hopf(
      double  _a  =  1.0,
      double  _b  =  1.0,
      double  _c  =  0.0
    ) :  a( _a ) , b( _b ) , c( _c )   { }

    void set_a   ( double _ )   { a = _; }
    void set_b   ( double _ )   { b = _; }
    void set_c   ( double _ )   { c = _; }

  };  // class general_hopf


};   // namespace dynamical_system
};   // namespace boost
};   // namespace numeric


#endif

