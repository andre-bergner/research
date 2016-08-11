/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef		_ADDITIVE_COUPLEABLE_
#define		_ADDITIVE_COUPLEABLE_

#include	"ode.hpp"

//#include <utility>


namespace boost {
namespace numeric {
namespace odeint {

struct linear_coupling
{
   double  weight;

   template < class state , class ftype >
   state  operator()  ( const state& x , const state& y , const ftype t ) {
      return  weight * y;
   }

};

};   // namespace dynamical_system
};   // namespace numeric
};   // namespace boost

#endif

